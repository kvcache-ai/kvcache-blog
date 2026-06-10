#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::size_t kChunkEvents = 1'000'000;

struct Options {
  std::string policy;
  std::string ids_path;
  std::string tokens_path;
  std::string next_path;
  std::string request_ends_path;
  std::uint64_t total_blocks = 0;
  std::uint64_t warmup_event_start = 0;
  std::uint64_t request_count = 0;
  std::uint64_t warmup_requests = 0;
  std::uint32_t capacity = 0;
};

struct Result {
  std::uint64_t hit_tokens = 0;
  std::uint64_t total_tokens = 0;
  std::uint64_t useful_cache_block_samples = 0;
  std::uint64_t useful_cache_samples = 0;

  double hit_rate() const {
    return total_tokens == 0 ? 0.0 : static_cast<double>(hit_tokens) / static_cast<double>(total_tokens);
  }

  double useful_cache_rate(std::uint64_t capacity) const {
    return capacity == 0 || useful_cache_samples == 0
      ? 0.0
      : static_cast<double>(useful_cache_block_samples) / static_cast<double>(useful_cache_samples * capacity);
  }
};

std::uint64_t parse_u64(const char* value, const std::string& name) {
  if (!value || *value == '\0') {
    throw std::runtime_error("Invalid integer for " + name);
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end && *end != '\0') {
    throw std::runtime_error("Invalid integer for " + name);
  }
  return static_cast<std::uint64_t>(parsed);
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 1) {
    const std::string arg = argv[index];
    auto require_value = [&](const std::string& name) -> const char* {
      if (index + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++index];
    };
    if (arg == "--policy") options.policy = require_value(arg);
    else if (arg == "--ids") options.ids_path = require_value(arg);
    else if (arg == "--tokens") options.tokens_path = require_value(arg);
    else if (arg == "--next") options.next_path = require_value(arg);
    else if (arg == "--request-ends") options.request_ends_path = require_value(arg);
    else if (arg == "--total-blocks") options.total_blocks = parse_u64(require_value(arg), arg);
    else if (arg == "--warmup-event-start") options.warmup_event_start = parse_u64(require_value(arg), arg);
    else if (arg == "--request-count") options.request_count = parse_u64(require_value(arg), arg);
    else if (arg == "--warmup-requests") options.warmup_requests = parse_u64(require_value(arg), arg);
    else if (arg == "--capacity") options.capacity = static_cast<std::uint32_t>(parse_u64(require_value(arg), arg));
    else throw std::runtime_error("Unknown argument: " + arg);
  }
  if (options.policy.empty() || options.ids_path.empty() || options.tokens_path.empty() || options.total_blocks == 0) {
    throw std::runtime_error("Usage: kv-cache-lab-native-sim --policy fifo|lru|optimal --ids PATH --tokens PATH --total-blocks N --warmup-event-start N --capacity N [--next PATH]");
  }
  if (options.policy != "build-next" && options.next_path.empty()) {
    throw std::runtime_error("--next is required");
  }
  if (options.policy != "build-next" && (options.request_ends_path.empty() || options.request_count == 0)) {
    throw std::runtime_error("--request-ends and --request-count are required");
  }
  return options;
}

std::vector<std::uint32_t> load_request_ends(const Options& options) {
  std::vector<std::uint32_t> request_ends(options.request_count);
  std::ifstream in(options.request_ends_path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open request-ends file");
  in.read(reinterpret_cast<char*>(request_ends.data()), static_cast<std::streamsize>(request_ends.size() * sizeof(std::uint32_t)));
  if (in.gcount() != static_cast<std::streamsize>(request_ends.size() * sizeof(std::uint32_t))) {
    throw std::runtime_error("Short read while reading request-ends file");
  }
  return request_ends;
}

struct RequestSampler {
  const Options& options;
  const std::vector<std::uint32_t>& request_ends;
  std::uint64_t request_index = 0;
  std::uint64_t useful_sum = 0;
  std::uint64_t samples = 0;

  void sample(std::uint64_t event_index, std::uint64_t useful_count) {
    const std::uint64_t event_end = event_index + 1;
    while (request_index < request_ends.size() && event_end >= request_ends[request_index]) {
      if (request_index >= options.warmup_requests) {
        useful_sum += useful_count;
        samples += 1;
      }
      request_index += 1;
    }
  }

  void fill(Result& result) const {
    result.useful_cache_block_samples = useful_sum;
    result.useful_cache_samples = samples;
  }
};

template <typename ChunkFn>
void scan_chunks(const Options& options, bool with_next, ChunkFn&& fn) {
  std::ifstream ids(options.ids_path, std::ios::binary);
  std::ifstream tokens(options.tokens_path, std::ios::binary);
  std::ifstream next;
  if (with_next) next.open(options.next_path, std::ios::binary);
  if (!ids) throw std::runtime_error("Failed to open ids file");
  if (!tokens) throw std::runtime_error("Failed to open tokens file");
  if (with_next && !next) throw std::runtime_error("Failed to open next file");

  std::vector<std::uint32_t> id_buffer(kChunkEvents);
  std::vector<std::uint16_t> token_buffer(kChunkEvents);
  std::vector<std::uint32_t> next_buffer(with_next ? kChunkEvents : 0);

  for (std::uint64_t start = 0; start < options.total_blocks; start += kChunkEvents) {
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(kChunkEvents, options.total_blocks - start));
    ids.read(reinterpret_cast<char*>(id_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    tokens.read(reinterpret_cast<char*>(token_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
    if (with_next) {
      next.read(reinterpret_cast<char*>(next_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    }
    if (ids.gcount() != static_cast<std::streamsize>(count * sizeof(std::uint32_t)) ||
        tokens.gcount() != static_cast<std::streamsize>(count * sizeof(std::uint16_t)) ||
        (with_next && next.gcount() != static_cast<std::streamsize>(count * sizeof(std::uint32_t)))) {
      throw std::runtime_error("Short read while scanning event stream");
    }
    fn(start, count, id_buffer.data(), token_buffer.data(), with_next ? next_buffer.data() : nullptr);
  }
}

void build_next_file(const Options& options) {
  if (options.next_path.empty()) throw std::runtime_error("--next is required for build-next");
  if (options.total_blocks + 1ULL > 0xffffffffULL) throw std::runtime_error("total blocks exceed uint32 next-use encoding");

  std::ifstream ids(options.ids_path, std::ios::binary);
  std::fstream next(options.next_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!ids) throw std::runtime_error("Failed to open ids file");
  if (!next) throw std::runtime_error("Failed to open next file");

  next.seekp(static_cast<std::streamoff>(options.total_blocks * sizeof(std::uint32_t) - 1));
  const char zero = 0;
  next.write(&zero, 1);
  next.flush();

  std::unordered_map<std::uint32_t, std::uint32_t> last_seen;
  std::vector<std::uint32_t> id_buffer(kChunkEvents);
  std::vector<std::uint32_t> next_buffer(kChunkEvents);
  const std::uint32_t never = static_cast<std::uint32_t>(options.total_blocks + 1ULL);

  for (std::uint64_t end = options.total_blocks; end > 0; end -= std::min<std::uint64_t>(kChunkEvents, end)) {
    const std::uint64_t start = end > kChunkEvents ? end - kChunkEvents : 0;
    const std::size_t count = static_cast<std::size_t>(end - start);
    ids.seekg(static_cast<std::streamoff>(start * sizeof(std::uint32_t)));
    ids.read(reinterpret_cast<char*>(id_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    if (ids.gcount() != static_cast<std::streamsize>(count * sizeof(std::uint32_t))) {
      throw std::runtime_error("Short read while building next-use file");
    }
    for (std::size_t reverse = count; reverse > 0; reverse -= 1) {
      const std::size_t index = reverse - 1;
      const std::uint32_t id = id_buffer[index];
      const auto found = last_seen.find(id);
      next_buffer[index] = found == last_seen.end() ? never : found->second;
      last_seen[id] = static_cast<std::uint32_t>(start + index);
    }
    next.seekp(static_cast<std::streamoff>(start * sizeof(std::uint32_t)));
    next.write(reinterpret_cast<const char*>(next_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
  }
}

Result simulate_fifo(const Options& options) {
  Result result;
  if (options.capacity == 0) return result;

  const std::vector<std::uint32_t> request_ends = load_request_ends(options);
  RequestSampler sampler{options, request_ends};
  const std::uint32_t never = static_cast<std::uint32_t>(options.total_blocks + 1ULL);
  std::unordered_map<std::uint32_t, std::uint32_t> cache;
  cache.reserve(options.capacity * 2ULL + 1ULL);
  std::vector<std::uint32_t> queue;
  queue.reserve(std::min<std::uint64_t>(options.total_blocks, static_cast<std::uint64_t>(options.capacity) * 4ULL + 1024ULL));
  std::size_t head = 0;
  std::uint64_t useful_count = 0;

  scan_chunks(options, true, [&](std::uint64_t start, std::size_t count, const std::uint32_t* ids, const std::uint16_t* tokens, const std::uint32_t* next_values) {
    for (std::size_t index = 0; index < count; index += 1) {
      const std::uint64_t event_index = start + index;
      const std::uint32_t id = ids[index];
      const std::uint16_t token_count = tokens[index];
      const std::uint32_t next_use = next_values[index];
      const auto found = cache.find(id);
      const bool hit = found != cache.end();
      if (event_index >= options.warmup_event_start) {
        result.total_tokens += token_count;
        if (hit) result.hit_tokens += token_count;
      }
      if (!hit) {
        while (cache.size() >= options.capacity && head < queue.size()) {
          const std::uint32_t victim = queue[head++];
          const auto victim_found = cache.find(victim);
          if (victim_found != cache.end()) {
            if (victim_found->second < never) useful_count -= 1;
            cache.erase(victim_found);
            break;
          }
        }
        if (cache.size() < options.capacity) {
          if (next_use < never) useful_count += 1;
          cache[id] = next_use;
          queue.push_back(id);
        }
      } else {
        if (found->second < never) useful_count -= 1;
        if (next_use < never) useful_count += 1;
        found->second = next_use;
      }
      sampler.sample(event_index, useful_count);
      if (head > 1'000'000 && head * 2 > queue.size()) {
        queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
      }
    }
  });
  sampler.fill(result);
  return result;
}

Result simulate_ceiling(const Options& options) {
  Result result;
  const std::vector<std::uint32_t> request_ends = load_request_ends(options);
  RequestSampler sampler{options, request_ends};
  const std::uint32_t never = static_cast<std::uint32_t>(options.total_blocks + 1ULL);
  std::unordered_map<std::uint32_t, std::uint32_t> seen;
  seen.reserve(static_cast<std::size_t>(options.capacity) + 1ULL);
  std::uint64_t useful_count = 0;
  scan_chunks(options, true, [&](std::uint64_t start, std::size_t count, const std::uint32_t* ids, const std::uint16_t* tokens, const std::uint32_t* next_values) {
    for (std::size_t index = 0; index < count; index += 1) {
      const std::uint64_t event_index = start + index;
      const std::uint32_t id = ids[index];
      const std::uint16_t token_count = tokens[index];
      const std::uint32_t next_use = next_values[index];
      const auto found = seen.find(id);
      const bool hit = found != seen.end();
      if (event_index >= options.warmup_event_start) {
        result.total_tokens += token_count;
        if (hit) result.hit_tokens += token_count;
      }
      if (hit && found->second < never) useful_count -= 1;
      if (next_use < never) useful_count += 1;
      seen[id] = next_use;
      sampler.sample(event_index, useful_count);
    }
  });
  sampler.fill(result);
  return result;
}

Result simulate_lru(const Options& options) {
  Result result;
  if (options.capacity == 0) return result;

  const std::vector<std::uint32_t> request_ends = load_request_ends(options);
  RequestSampler sampler{options, request_ends};
  const std::uint32_t never = static_cast<std::uint32_t>(options.total_blocks + 1ULL);
  std::list<std::uint32_t> order;
  struct LruEntry {
    std::list<std::uint32_t>::iterator it;
    std::uint32_t next_use = 0;
  };
  std::unordered_map<std::uint32_t, LruEntry> cache;
  cache.reserve(options.capacity * 2ULL + 1ULL);
  std::uint64_t useful_count = 0;

  scan_chunks(options, true, [&](std::uint64_t start, std::size_t count, const std::uint32_t* ids, const std::uint16_t* tokens, const std::uint32_t* next_values) {
    for (std::size_t index = 0; index < count; index += 1) {
      const std::uint64_t event_index = start + index;
      const std::uint32_t id = ids[index];
      const std::uint16_t token_count = tokens[index];
      const std::uint32_t next_use = next_values[index];
      const auto found = cache.find(id);
      const bool hit = found != cache.end();
      if (event_index >= options.warmup_event_start) {
        result.total_tokens += token_count;
        if (hit) result.hit_tokens += token_count;
      }
      if (hit) {
        if (found->second.next_use < never) useful_count -= 1;
        order.splice(order.end(), order, found->second.it);
        found->second.it = std::prev(order.end());
        found->second.next_use = next_use;
      } else {
        while (cache.size() >= options.capacity && !order.empty()) {
          const std::uint32_t victim = order.front();
          order.pop_front();
          const auto victim_found = cache.find(victim);
          if (victim_found != cache.end() && victim_found->second.next_use < never) useful_count -= 1;
          cache.erase(victim);
        }
        if (cache.size() < options.capacity) {
          order.push_back(id);
          cache[id] = {std::prev(order.end()), next_use};
        }
      }
      if (next_use < never) useful_count += 1;
      sampler.sample(event_index, useful_count);
    }
  });
  sampler.fill(result);
  return result;
}

struct OptimalState {
  std::uint32_t next_use = 0;
  std::uint32_t version = 0;
};

struct HeapEntry {
  std::uint32_t next_use = 0;
  std::uint32_t id = 0;
  std::uint32_t version = 0;
  bool operator<(const HeapEntry& other) const {
    return next_use < other.next_use;
  }
};

Result simulate_optimal(const Options& options) {
  Result result;
  if (options.capacity == 0) return result;

  const std::vector<std::uint32_t> request_ends = load_request_ends(options);
  RequestSampler sampler{options, request_ends};
  const std::uint32_t never = static_cast<std::uint32_t>(options.total_blocks + 1ULL);
  std::unordered_map<std::uint32_t, OptimalState> cache;
  cache.reserve(options.capacity * 2ULL + 1ULL);
  std::priority_queue<HeapEntry> heap;
  std::uint64_t useful_count = 0;

  auto rebuild_heap = [&]() {
    std::priority_queue<HeapEntry> rebuilt;
    for (const auto& item : cache) {
      rebuilt.push({ item.second.next_use, item.first, item.second.version });
    }
    heap.swap(rebuilt);
  };

  auto push_state = [&](std::uint32_t id, std::uint32_t next_use) {
    auto& state = cache[id];
    if (state.version > 0 && state.next_use < never) useful_count -= 1;
    if (next_use < never) useful_count += 1;
    state.next_use = next_use;
    state.version += 1;
    heap.push({ next_use, id, state.version });
  };

  auto evict_for_candidate = [&](std::uint32_t candidate_next) {
    for (;;) {
      if (heap.empty()) {
        rebuild_heap();
        if (heap.empty()) return cache.size() < options.capacity;
      }
      const HeapEntry top = heap.top();
      heap.pop();
      const auto found = cache.find(top.id);
      if (found == cache.end()) continue;
      if (found->second.version != top.version || found->second.next_use != top.next_use) continue;
      if (found->second.next_use > candidate_next) {
        if (found->second.next_use < never) useful_count -= 1;
        cache.erase(found);
        return true;
      }
      heap.push(top);
      return false;
    }
  };

  scan_chunks(options, true, [&](std::uint64_t start, std::size_t count, const std::uint32_t* ids, const std::uint16_t* tokens, const std::uint32_t* next_values) {
    for (std::size_t index = 0; index < count; index += 1) {
      const std::uint64_t event_index = start + index;
      const std::uint32_t id = ids[index];
      const std::uint16_t token_count = tokens[index];
      const std::uint32_t next_use = next_values[index];
      const bool hit = cache.find(id) != cache.end();
      if (event_index >= options.warmup_event_start) {
        result.total_tokens += token_count;
        if (hit) result.hit_tokens += token_count;
      }
      if (hit) {
        push_state(id, next_use);
      } else if (cache.size() < options.capacity) {
        push_state(id, next_use);
      } else if (evict_for_candidate(next_use)) {
        if (cache.size() < options.capacity) push_state(id, next_use);
      } else {
        // Belady-with-bypass: keep the current cache if the miss would be used
        // no sooner than every resident block.
      }
      sampler.sample(event_index, useful_count);
      if (heap.size() > cache.size() * 2ULL + 1'000'000ULL) rebuild_heap();
    }
  });
  sampler.fill(result);
  return result;
}

void print_result(const Options& options, const Result& result) {
  std::cout << "{"
            << "\"policy\":\"" << options.policy << "\","
            << "\"cacheBlocks\":" << options.capacity << ","
            << "\"hitTokens\":" << result.hit_tokens << ","
            << "\"totalTokens\":" << result.total_tokens << ","
            << "\"hitRate\":" << std::setprecision(17) << result.hit_rate() << ","
            << "\"usefulCacheBlockSamples\":" << result.useful_cache_block_samples << ","
            << "\"usefulCacheSamples\":" << result.useful_cache_samples << ","
            << "\"usefulCacheRate\":" << std::setprecision(17) << result.useful_cache_rate(options.capacity)
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    if (options.policy == "build-next") {
      build_next_file(options);
      std::cout << "{\"policy\":\"build-next\",\"cacheBlocks\":0,\"hitTokens\":0,\"totalTokens\":0,\"hitRate\":0}\n";
    }
    else if (options.policy == "ceiling") print_result(options, simulate_ceiling(options));
    else if (options.policy == "fifo") print_result(options, simulate_fifo(options));
    else if (options.policy == "lru") print_result(options, simulate_lru(options));
    else if (options.policy == "optimal") print_result(options, simulate_optimal(options));
    else throw std::runtime_error("Unsupported policy: " + options.policy);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
