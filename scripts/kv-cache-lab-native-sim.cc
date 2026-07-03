#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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
  std::int64_t measurement_start_request = -1;

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
  if (!value || *value == '\0') throw std::runtime_error("Invalid integer for " + name);
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end && *end != '\0') throw std::runtime_error("Invalid integer for " + name);
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
    throw std::runtime_error("Usage: kv-cache-lab-native-sim --policy fifo|lru|optimal|all|ceiling --ids PATH --tokens PATH --total-blocks N --request-ends PATH --request-count N --capacity N [--next PATH]");
  }
  if (options.policy != "build-next" && (options.request_ends_path.empty() || options.request_count == 0)) {
    throw std::runtime_error("--request-ends and --request-count are required");
  }
  if (options.policy != "build-next" && options.next_path.empty()) {
    throw std::runtime_error("--next is required");
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

std::vector<std::uint32_t> load_ids_all(const Options& options) {
  std::vector<std::uint32_t> ids(options.total_blocks);
  std::ifstream in(options.ids_path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open ids file");
  in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(std::uint32_t)));
  if (in.gcount() != static_cast<std::streamsize>(ids.size() * sizeof(std::uint32_t))) {
    throw std::runtime_error("Short read while reading ids file");
  }
  return ids;
}

std::vector<std::uint16_t> load_tokens_all(const Options& options) {
  std::vector<std::uint16_t> tokens(options.total_blocks);
  std::ifstream in(options.tokens_path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open tokens file");
  in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(std::uint16_t)));
  if (in.gcount() != static_cast<std::streamsize>(tokens.size() * sizeof(std::uint16_t))) {
    throw std::runtime_error("Short read while reading tokens file");
  }
  return tokens;
}

template <typename ChunkFn>
void scan_chunks(const Options& options, ChunkFn&& fn) {
  std::ifstream ids(options.ids_path, std::ios::binary);
  if (!ids) throw std::runtime_error("Failed to open ids file");
  std::vector<std::uint32_t> id_buffer(kChunkEvents);
  for (std::uint64_t start = 0; start < options.total_blocks; start += kChunkEvents) {
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(kChunkEvents, options.total_blocks - start));
    ids.read(reinterpret_cast<char*>(id_buffer.data()), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
    if (ids.gcount() != static_cast<std::streamsize>(count * sizeof(std::uint32_t))) {
      throw std::runtime_error("Short read while scanning event stream");
    }
    fn(start, count, id_buffer.data());
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

struct PrefixPlan {
  std::vector<std::uint32_t> node_for_event;
  std::vector<std::uint32_t> next_request_for_event;
  std::vector<std::uint32_t> parent;
};

struct SimulationInput {
  std::vector<std::uint32_t> ids;
  std::vector<std::uint16_t> tokens;
  std::vector<std::uint32_t> request_ends;
  PrefixPlan prefix;
  std::uint32_t unique_blocks = 0;
};

PrefixPlan build_prefix_plan(const std::vector<std::uint32_t>& ids,
                             const std::vector<std::uint32_t>& request_ends) {
  PrefixPlan plan;
  plan.node_for_event.resize(ids.size());
  plan.next_request_for_event.resize(ids.size());
  plan.parent.push_back(0);

  std::unordered_map<std::uint64_t, std::uint32_t> edge_to_node;
  edge_to_node.reserve(std::min<std::size_t>(ids.size(), 8'000'000ULL));

  std::uint64_t start = 0;
  for (std::uint32_t end : request_ends) {
    std::uint32_t parent = 0;
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint64_t key = (static_cast<std::uint64_t>(parent) << 32) | ids[index];
      auto found = edge_to_node.find(key);
      std::uint32_t node = 0;
      if (found == edge_to_node.end()) {
        node = static_cast<std::uint32_t>(plan.parent.size());
        edge_to_node.emplace(key, node);
        plan.parent.push_back(parent);
      } else {
        node = found->second;
      }
      plan.node_for_event[index] = node;
      parent = node;
    }
    start = end;
  }

  const std::uint32_t never_request = static_cast<std::uint32_t>(request_ends.size() + 1ULL);
  std::vector<std::uint32_t> last_use(plan.parent.size(), never_request);
  for (std::uint64_t reverse = request_ends.size(); reverse > 0; reverse -= 1) {
    const std::uint64_t request = reverse - 1;
    const std::uint64_t request_start = request == 0 ? 0 : request_ends[request - 1];
    const std::uint64_t request_end = request_ends[request];
    for (std::uint64_t index = request_start; index < request_end; index += 1) {
      plan.next_request_for_event[index] = last_use[plan.node_for_event[index]];
    }
    for (std::uint64_t index = request_start; index < request_end; index += 1) {
      last_use[plan.node_for_event[index]] = static_cast<std::uint32_t>(request);
    }
  }
  return plan;
}

SimulationInput load_simulation_input(const Options& options) {
  SimulationInput input;
  input.ids = load_ids_all(options);
  input.tokens = load_tokens_all(options);
  input.request_ends = load_request_ends(options);
  input.prefix = build_prefix_plan(input.ids, input.request_ends);
  input.unique_blocks = static_cast<std::uint32_t>(input.prefix.parent.size() - 1);
  return input;
}

std::uint64_t total_tokens(const SimulationInput& input, std::uint64_t start_request = 0) {
  std::uint64_t total = 0;
  const std::uint64_t request_count = input.request_ends.size();
  for (std::uint64_t request = std::min(start_request, request_count); request < request_count; request += 1) {
    const std::uint64_t start = request == 0 ? 0 : input.request_ends[request - 1];
    const std::uint64_t end = input.request_ends[request];
    for (std::uint64_t index = start; index < end; index += 1) total += input.tokens[index];
  }
  return total;
}

Result no_cache_result(const SimulationInput& input, const Options& options) {
  Result result;
  result.total_tokens = total_tokens(input, options.warmup_requests);
  result.measurement_start_request = static_cast<std::int64_t>(options.warmup_requests);
  return result;
}

Result underfilled_result(const SimulationInput& input, const Options& options) {
  Result result;
  result.total_tokens = total_tokens(input, options.warmup_requests);
  result.measurement_start_request = -2;
  return result;
}

Result simulate_ceiling(const SimulationInput& input, const Options& options) {
  Result result;
  std::vector<std::uint8_t> seen(input.prefix.parent.size(), 0);
  std::uint64_t start = 0;
  for (std::uint64_t request = 0; request < input.request_ends.size(); request += 1) {
    const std::uint64_t end = input.request_ends[request];
    const bool measured = request >= options.warmup_requests;
    bool prefix_alive = true;
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t node = input.prefix.node_for_event[index];
      const bool hit = seen[node] != 0;
      if (measured) result.total_tokens += input.tokens[index];
      if (measured && prefix_alive && hit) {
        result.hit_tokens += input.tokens[index];
      } else if (!hit) {
        prefix_alive = false;
      }
    }
    for (std::uint64_t index = start; index < end; index += 1) {
      seen[input.prefix.node_for_event[index]] = 1;
    }
    start = end;
  }
  result.measurement_start_request = static_cast<std::int64_t>(options.warmup_requests);
  return result;
}

Result simulate_fifo(const SimulationInput& input, std::uint32_t capacity, const Options& options) {
  if (capacity == 0 || input.ids.empty()) return no_cache_result(input, options);
  Result result;
  std::vector<std::uint8_t> in_cache(input.prefix.parent.size(), 0);
  std::vector<std::uint32_t> queue;
  queue.reserve(std::min<std::size_t>(input.ids.size(), static_cast<std::size_t>(capacity) * 4ULL + 1024ULL));
  std::size_t head = 0;
  std::uint32_t cache_size = 0;
  bool full_before_measurement = false;

  std::uint64_t start = 0;
  for (std::uint64_t request = 0; request < input.request_ends.size(); request += 1) {
    if (request >= options.warmup_requests && !full_before_measurement) return underfilled_result(input, options);
    const std::uint64_t end = input.request_ends[request];
    const bool measured = request >= options.warmup_requests;
    bool prefix_alive = true;
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t id = input.prefix.node_for_event[index];
      const bool hit = in_cache[id] != 0;
      if (measured) {
        result.total_tokens += input.tokens[index];
        if (prefix_alive && hit) result.hit_tokens += input.tokens[index];
      }
      if (!hit) prefix_alive = false;
    }
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t id = input.prefix.node_for_event[index];
      if (in_cache[id]) continue;
      if (cache_size >= capacity) {
        while (head < queue.size()) {
          const std::uint32_t victim = queue[head++];
          if (in_cache[victim]) {
            in_cache[victim] = 0;
            cache_size -= 1;
            break;
          }
        }
      }
      if (cache_size < capacity) {
        in_cache[id] = 1;
        cache_size += 1;
        queue.push_back(id);
        if (cache_size >= capacity && request < options.warmup_requests) full_before_measurement = true;
      }
      if (head > 1'000'000 && head * 2 > queue.size()) {
        queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
      }
    }
    start = end;
  }
  if (!full_before_measurement || result.total_tokens == 0) return underfilled_result(input, options);
  result.measurement_start_request = static_cast<std::int64_t>(options.warmup_requests);
  return result;
}

struct HeapItem {
  std::uint32_t node = 0;
  std::uint32_t key = 0;
  std::uint32_t version = 0;
};

class BinaryHeap {
 public:
  explicit BinaryHeap(bool max_heap) : max_heap_(max_heap) {}

  void push(const HeapItem& item) {
    items_.push_back(item);
    std::size_t index = items_.size() - 1;
    while (index > 0) {
      const std::size_t parent = (index - 1) >> 1;
      if (!better(items_[index], items_[parent])) break;
      std::swap(items_[index], items_[parent]);
      index = parent;
    }
  }

  bool pop(HeapItem& out) {
    if (items_.empty()) return false;
    out = items_[0];
    const HeapItem last = items_.back();
    items_.pop_back();
    if (!items_.empty()) {
      items_[0] = last;
      std::size_t index = 0;
      for (;;) {
        const std::size_t left = index * 2 + 1;
        const std::size_t right = left + 1;
        std::size_t best = index;
        if (left < items_.size() && better(items_[left], items_[best])) best = left;
        if (right < items_.size() && better(items_[right], items_[best])) best = right;
        if (best == index) break;
        std::swap(items_[best], items_[index]);
        index = best;
      }
    }
    return true;
  }

 private:
  bool better(const HeapItem& left, const HeapItem& right) const {
    if (left.key == right.key) return left.node > right.node;
    return max_heap_ ? left.key > right.key : left.key < right.key;
  }

  bool max_heap_ = false;
  std::vector<HeapItem> items_;
};

Result simulate_trie_policy(const SimulationInput& input, std::uint32_t capacity, bool optimal, const Options& options) {
  if (capacity == 0 || input.ids.empty()) return no_cache_result(input, options);

  Result result;
  const std::size_t node_count = input.prefix.parent.size();
  std::vector<std::uint8_t> present(node_count, 0);
  std::vector<std::uint32_t> child_count(node_count, 0);
  std::vector<std::uint32_t> state_version(node_count, 0);
  std::vector<std::uint32_t> state_key(node_count, 0);
  std::vector<std::uint32_t> protected_mark(node_count, 0);
  BinaryHeap heap(optimal);
  present[0] = 1;
  std::uint32_t cache_size = 0;
  std::uint32_t clock = 0;
  std::uint32_t mark_value = 1;
  bool full_before_measurement = false;

  auto push_leaf = [&](std::uint32_t node) {
    if (node == 0 || !present[node] || child_count[node] != 0) return;
    state_version[node] += 1;
    heap.push({ node, state_key[node], state_version[node] });
  };

  auto touch_lru = [&](std::uint32_t node) {
    if (node == 0 || !present[node]) return;
    state_key[node] = ++clock;
    push_leaf(node);
  };

  auto update_optimal = [&](std::uint32_t node, std::uint32_t next_use) {
    if (node == 0 || !present[node]) return;
    state_key[node] = next_use;
    push_leaf(node);
  };

  auto add_node = [&](std::uint32_t node, std::uint64_t event_index) {
    const std::uint32_t parent = input.prefix.parent[node];
    present[node] = 1;
    cache_size += 1;
    child_count[parent] += 1;
    if (optimal) update_optimal(node, input.prefix.next_request_for_event[event_index]);
    else touch_lru(node);
  };

  auto evict_leaf = [&](std::uint32_t candidate_key) {
    std::vector<HeapItem> skipped;
    HeapItem top;
    for (;;) {
      if (!heap.pop(top)) {
        for (const HeapItem& item : skipped) heap.push(item);
        return false;
      }
      const std::uint32_t node = top.node;
      if (!present[node] || child_count[node] != 0 || state_version[node] != top.version || state_key[node] != top.key) {
        continue;
      }
      if (protected_mark[node] == mark_value) {
        skipped.push_back(top);
        continue;
      }
      if (optimal && top.key <= candidate_key) {
        heap.push(top);
        for (const HeapItem& item : skipped) heap.push(item);
        return false;
      }
      present[node] = 0;
      cache_size -= 1;
      const std::uint32_t parent = input.prefix.parent[node];
      child_count[parent] -= 1;
      if (parent > 0 && present[parent] && child_count[parent] == 0) push_leaf(parent);
      for (const HeapItem& item : skipped) heap.push(item);
      return true;
    }
  };

  auto mark_protected_path = [&](std::uint64_t start, std::uint64_t end) {
    mark_value += 1;
    if (mark_value == 0x7fffffffU) {
      std::fill(protected_mark.begin(), protected_mark.end(), 0);
      mark_value = 1;
    }
    protected_mark[0] = mark_value;
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t node = input.prefix.node_for_event[index];
      if (present[node]) protected_mark[node] = mark_value;
    }
  };

  std::uint64_t start = 0;
  for (std::uint64_t request = 0; request < input.request_ends.size(); request += 1) {
    if (request >= options.warmup_requests && !full_before_measurement) return underfilled_result(input, options);
    const std::uint64_t end = input.request_ends[request];
    const bool measured = request >= options.warmup_requests;
    bool prefix_alive = true;
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t node = input.prefix.node_for_event[index];
      const bool hit = present[node] != 0;
      if (measured) {
        result.total_tokens += input.tokens[index];
        if (prefix_alive && hit) result.hit_tokens += input.tokens[index];
      }
      if (prefix_alive && hit) {
        if (optimal) update_optimal(node, input.prefix.next_request_for_event[index]);
        else touch_lru(node);
      } else if (!hit) {
        prefix_alive = false;
      }
    }

    mark_protected_path(start, end);
    for (std::uint64_t index = start; index < end; index += 1) {
      const std::uint32_t node = input.prefix.node_for_event[index];
      if (present[node]) continue;
      if (cache_size >= capacity) {
        const std::uint32_t candidate_key = optimal ? input.prefix.next_request_for_event[index] : 0;
        if (!evict_leaf(candidate_key)) break;
      }
      if (cache_size < capacity && present[input.prefix.parent[node]]) {
        add_node(node, index);
        if (cache_size >= capacity && request < options.warmup_requests) full_before_measurement = true;
      } else {
        break;
      }
    }
    start = end;
  }
  if (!full_before_measurement || result.total_tokens == 0) return underfilled_result(input, options);
  result.measurement_start_request = static_cast<std::int64_t>(options.warmup_requests);
  return result;
}

struct AllResults {
  Result fifo;
  Result lru;
  Result optimal;
};

AllResults simulate_all(const SimulationInput& input, std::uint32_t capacity, const Options& options) {
  return {
    simulate_fifo(input, capacity, options),
    simulate_trie_policy(input, capacity, false, options),
    simulate_trie_policy(input, capacity, true, options),
  };
}

void print_result(const std::string& policy, std::uint64_t cache_blocks, const Result& result, std::uint64_t trie_node_count, std::uint64_t warmup_requests) {
  std::cout << "{"
            << "\"policy\":\"" << policy << "\","
            << "\"cacheBlocks\":" << cache_blocks << ","
            << "\"trieNodeCount\":" << trie_node_count << ","
            << "\"warmupRequests\":" << warmup_requests << ","
            << "\"measurementStartRequest\":" << result.measurement_start_request << ","
            << "\"hitTokens\":" << result.hit_tokens << ","
            << "\"totalTokens\":" << result.total_tokens << ","
            << "\"hitRate\":" << std::setprecision(17) << result.hit_rate() << ","
            << "\"usefulCacheBlockSamples\":" << result.useful_cache_block_samples << ","
            << "\"usefulCacheSamples\":" << result.useful_cache_samples << ","
            << "\"usefulCacheRate\":" << std::setprecision(17) << result.useful_cache_rate(cache_blocks)
            << "}\n";
}

void print_named_result(const char* policy, std::uint64_t cache_blocks, const Result& result, std::uint64_t warmup_requests) {
  std::cout << "\"policy\":\"" << policy << "\","
            << "\"cacheBlocks\":" << cache_blocks << ","
            << "\"warmupRequests\":" << warmup_requests << ","
            << "\"measurementStartRequest\":" << result.measurement_start_request << ","
            << "\"hitTokens\":" << result.hit_tokens << ","
            << "\"totalTokens\":" << result.total_tokens << ","
            << "\"hitRate\":" << std::setprecision(17) << result.hit_rate() << ","
            << "\"usefulCacheBlockSamples\":" << result.useful_cache_block_samples << ","
            << "\"usefulCacheSamples\":" << result.useful_cache_samples << ","
            << "\"usefulCacheRate\":" << std::setprecision(17) << result.useful_cache_rate(cache_blocks);
}

void print_all_results(std::uint64_t cache_blocks, const AllResults& results, std::uint64_t trie_node_count, std::uint64_t warmup_requests) {
  std::cout << "{"
            << "\"cacheBlocks\":" << cache_blocks << ","
            << "\"trieNodeCount\":" << trie_node_count << ","
            << "\"fifo\":{";
  print_named_result("fifo", cache_blocks, results.fifo, warmup_requests);
  std::cout << "},\"lru\":{";
  print_named_result("lru", cache_blocks, results.lru, warmup_requests);
  std::cout << "},\"optimal\":{";
  print_named_result("optimal", cache_blocks, results.optimal, warmup_requests);
  std::cout << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    if (options.policy == "build-next") {
      build_next_file(options);
      std::cout << "{\"policy\":\"build-next\",\"cacheBlocks\":0,\"hitTokens\":0,\"totalTokens\":0,\"hitRate\":0}\n";
      return 0;
    }

    const SimulationInput input = load_simulation_input(options);
    const std::uint64_t trie_node_count = input.prefix.parent.size();
    if (options.policy == "ceiling") print_result("ceiling", options.capacity, simulate_ceiling(input, options), trie_node_count, options.warmup_requests);
    else if (options.policy == "fifo") print_result("fifo", options.capacity, simulate_fifo(input, options.capacity, options), trie_node_count, options.warmup_requests);
    else if (options.policy == "lru") print_result("lru", options.capacity, simulate_trie_policy(input, options.capacity, false, options), trie_node_count, options.warmup_requests);
    else if (options.policy == "optimal") print_result("optimal", options.capacity, simulate_trie_policy(input, options.capacity, true, options), trie_node_count, options.warmup_requests);
    else if (options.policy == "all") print_all_results(options.capacity, simulate_all(input, options.capacity, options), trie_node_count, options.warmup_requests);
    else throw std::runtime_error("Unsupported policy: " + options.policy);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
