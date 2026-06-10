// KV Cache Lab WASM trace processor.
//
// JS streams a (gzip-decompressed) JSONL file and feeds raw UTF-8 byte chunks to
// `ingest`; this module parses each line, interns the 64-bit `hash_ids` exactly,
// and builds the interned event-stream "plan" (ids + per-block token weights +
// per-request offsets/timestamps) directly in WASM linear memory. After
// `finalize` it answers `sweep(cacheBlocks, policy)` for any capacity. Keeping
// the whole working set in one linear memory — and freeing the interner before
// the sweep — lets a multi-GB / hundreds-of-millions-of-events trace load fully
// in the browser, which the JS-heap path cannot.
//
// The FIFO / LRU / Optimal (Belady-with-bypass) kernels, the next-use build, the
// warmup boundary, and the infinite-cache ceiling are ported 1:1 from the
// JS implementation in assets/js/kv-cache-lab.js, so results are identical.
#![allow(static_mut_refs)]

const POLICY_FIFO: u32 = 0;
const POLICY_LRU: u32 = 1;
const POLICY_OPTIMAL: u32 = 2;

// Open-addressing u64 -> dense-id map. Dense ids are assigned in first-seen
// order, exactly like the JS `idMap.size` interner; ~12 B/slot keeps a
// tens-of-millions-of-uniques trace affordable, and it is dropped before the
// sweep so it never competes with the plan for memory.
struct Interner {
    keys: Vec<u64>,
    ids: Vec<u32>,
    mask: usize,
    size: u32,
}

const EMPTY: u32 = u32::MAX;

impl Interner {
    fn new(cap_hint: usize) -> Self {
        let mut cap = 16usize;
        let target = cap_hint.max(16).saturating_mul(2);
        while cap < target {
            cap <<= 1;
        }
        Interner {
            keys: vec![0u64; cap],
            ids: vec![EMPTY; cap],
            mask: cap - 1,
            size: 0,
        }
    }

    #[inline]
    fn hash(v: u64) -> usize {
        let mut h = v;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h as usize
    }

    #[inline]
    fn intern(&mut self, v: u64) -> u32 {
        let mut i = Self::hash(v) & self.mask;
        loop {
            let id = self.ids[i];
            if id == EMPTY {
                let assigned = self.size;
                self.ids[i] = assigned;
                self.keys[i] = v;
                self.size += 1;
                if (self.size as usize) * 4 >= self.keys.len() * 3 {
                    self.grow();
                }
                return assigned;
            }
            if self.keys[i] == v {
                return id;
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let new_cap = self.keys.len() * 2;
        let mut nk = vec![0u64; new_cap];
        let mut ni = vec![EMPTY; new_cap];
        let mask = new_cap - 1;
        for j in 0..self.keys.len() {
            let id = self.ids[j];
            if id == EMPTY {
                continue;
            }
            let v = self.keys[j];
            let mut i = Self::hash(v) & mask;
            while ni[i] != EMPTY {
                i = (i + 1) & mask;
            }
            ni[i] = id;
            nk[i] = v;
        }
        self.keys = nk;
        self.ids = ni;
        self.mask = mask;
    }
}

struct State {
    block_size: u32, // resolved trace block size (0 until set; seeded from the explicit option)
    max_events: u32, // 0 = no cap (full load)
    warmup_fraction: f64,

    interner: Interner,
    ids: Vec<u32>,        // interned block id per event
    toks: Vec<u16>,       // token weight per event
    req_start: Vec<u32>,  // event offset at each request start (+ sentinel after finalize)
    ts: Vec<f64>,         // timestamp per request
    next_use: Vec<u32>,   // next-use event index per event (built in finalize)

    total_input_tokens: f64,
    parse_errors: u32,
    skipped: u32,
    missing_block_size: u32,
    inconsistent_block_size: u32,
    capped: bool,
    requests: u32,
    unique: u32,
    t_min: f64,
    t_max: f64,
    warmup_requests: u32,
    warmup_event_start: u32,
    total_measured: f64,
    ceiling_hit: f64,

    line: Vec<u8>,    // partial-line carry across chunks
    staging: Vec<u8>, // input byte buffer JS writes into
    line_ids: Vec<u64>, // scratch for the current line's hash_ids
    time_out: Vec<f64>, // packed temporal-statistics output for JS to read
    stats_out: Vec<f64>, // [hit_tokens, useful_block_sample_sum, useful_sample_count]
}

impl State {
    fn new(block_size_override: u32, max_events: u32, warmup_fraction: f64) -> Self {
        let unique_hint = if max_events > 0 { (max_events / 4) as usize } else { 1 << 16 };
        State {
            block_size: block_size_override,
            max_events,
            warmup_fraction,
            interner: Interner::new(unique_hint),
            ids: Vec::new(),
            toks: Vec::new(),
            req_start: Vec::new(),
            ts: Vec::new(),
            next_use: Vec::new(),
            total_input_tokens: 0.0,
            parse_errors: 0,
            skipped: 0,
            missing_block_size: 0,
            inconsistent_block_size: 0,
            capped: false,
            requests: 0,
            unique: 0,
            t_min: f64::INFINITY,
            t_max: f64::NEG_INFINITY,
            warmup_requests: 0,
            warmup_event_start: 0,
            total_measured: 0.0,
            ceiling_hit: 0.0,
            line: Vec::new(),
            staging: Vec::new(),
            line_ids: Vec::new(),
            time_out: Vec::new(),
            stats_out: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct SimStats {
    hit: f64,
    useful_sum: f64,
    samples: f64,
}

impl SimStats {
    fn empty() -> Self {
        SimStats {
            hit: 0.0,
            useful_sum: 0.0,
            samples: 0.0,
        }
    }
}

#[inline]
fn sample_request(s: &State, request_cursor: &mut usize, event_index: usize, useful_count: u32, stats: &mut SimStats) {
    let event_end = event_index + 1;
    let requests = s.requests as usize;
    while *request_cursor < requests && event_end >= s.req_start[*request_cursor + 1] as usize {
        if *request_cursor >= s.warmup_requests as usize {
            stats.useful_sum += useful_count as f64;
            stats.samples += 1.0;
        }
        *request_cursor += 1;
    }
}

static mut STATE: Option<State> = None;

#[inline]
fn st() -> &'static mut State {
    unsafe { STATE.as_mut().unwrap() }
}

// blockTokens() from kv-cache-lab.js, with input_length defaulted to 1 (matching
// toPositiveInteger(record.input_length, 1)) and block_size already resolved.
#[inline]
fn block_tokens(input_length: u64, block_size: u32, index: u32, count: u32) -> u16 {
    if count == 0 {
        return 0;
    }
    if block_size == 0 {
        // JS: Math.max(1, Math.round(inputLength / count))
        let r = ((input_length as f64) / (count as f64)).round();
        return r.max(1.0).min(65535.0) as u16;
    }
    let remaining = input_length as i64 - (index as i64) * (block_size as i64);
    if remaining <= 0 {
        return 1;
    }
    let r = remaining.min(block_size as i64).max(1);
    r.min(65535) as u16
}

#[inline]
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let first = needle[0];
    let last = hay.len() - needle.len();
    let mut i = 0;
    while i <= last {
        if hay[i] == first && &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

// Parse a number following `"key":` as u64 (integer part only).
fn find_number_u64(line: &[u8], key: &[u8]) -> Option<u64> {
    let pos = find(line, key)?;
    let mut i = pos + key.len();
    while i < line.len() && line[i] != b':' {
        i += 1;
    }
    i += 1;
    while i < line.len() && (line[i] == b' ' || line[i] == b'"') {
        i += 1;
    }
    let start = i;
    let mut v: u64 = 0;
    while i < line.len() && line[i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((line[i] - b'0') as u64);
        i += 1;
    }
    if i == start {
        None
    } else {
        Some(v)
    }
}

// Parse a number following `"key":` as f64 (handles decimals; ignores exponents,
// which timestamps do not use).
fn find_number_f64(line: &[u8], key: &[u8]) -> Option<f64> {
    let pos = find(line, key)?;
    let mut i = pos + key.len();
    while i < line.len() && line[i] != b':' {
        i += 1;
    }
    i += 1;
    while i < line.len() && (line[i] == b' ' || line[i] == b'"') {
        i += 1;
    }
    let start = i;
    let mut int_part: f64 = 0.0;
    while i < line.len() && line[i].is_ascii_digit() {
        int_part = int_part * 10.0 + (line[i] - b'0') as f64;
        i += 1;
    }
    let mut value = int_part;
    if i < line.len() && line[i] == b'.' {
        i += 1;
        let mut frac: f64 = 0.0;
        let mut scale: f64 = 1.0;
        while i < line.len() && line[i].is_ascii_digit() {
            frac = frac * 10.0 + (line[i] - b'0') as f64;
            scale *= 10.0;
            i += 1;
        }
        value += frac / scale;
    }
    if i == start {
        None
    } else {
        Some(value)
    }
}

fn parse_line(s: &mut State, line: &[u8]) {
    // trim ASCII whitespace
    let mut a = 0;
    let mut b = line.len();
    while a < b && line[a].is_ascii_whitespace() {
        a += 1;
    }
    while b > a && line[b - 1].is_ascii_whitespace() {
        b -= 1;
    }
    let line = &line[a..b];
    if line.is_empty() {
        return;
    }

    let hpos = match find(line, b"\"hash_ids\"") {
        Some(p) => p,
        None => {
            // Mirror JS JSON.parse: a JSON object without hash_ids is "skipped",
            // anything that isn't an object counts as a parse error.
            if line[0] == b'{' {
                s.skipped += 1;
            } else {
                s.parse_errors += 1;
            }
            return;
        }
    };

    // locate '[' after the key
    let mut i = hpos + 10;
    while i < line.len() && line[i] != b'[' {
        i += 1;
    }
    if i >= line.len() {
        s.skipped += 1;
        return;
    }
    i += 1;

    s.line_ids.clear();
    loop {
        while i < line.len() && !line[i].is_ascii_digit() && line[i] != b']' {
            i += 1;
        }
        if i >= line.len() || line[i] == b']' {
            break;
        }
        let mut v: u64 = 0;
        while i < line.len() && line[i].is_ascii_digit() {
            v = v.wrapping_mul(10).wrapping_add((line[i] - b'0') as u64);
            i += 1;
        }
        s.line_ids.push(v);
    }
    let count = s.line_ids.len() as u32;
    if count == 0 {
        s.skipped += 1;
        return;
    }

    let record_block_size = match find_number_u64(line, b"\"block_size\"")
        .map(|v| v as u32)
        .filter(|&v| v > 0)
    {
        Some(v) => v,
        None => {
            s.missing_block_size += 1;
            return;
        }
    };
    if s.block_size == 0 {
        s.block_size = record_block_size;
    } else if record_block_size != s.block_size {
        s.inconsistent_block_size += 1;
        return;
    }
    let input_length = find_number_u64(line, b"\"input_length\"").filter(|&v| v > 0).unwrap_or(1);
    let timestamp = find_number_f64(line, b"\"timestamp\"").unwrap_or(0.0);

    s.req_start.push(s.ids.len() as u32);
    s.ts.push(timestamp);
    if timestamp < s.t_min {
        s.t_min = timestamp;
    }
    if timestamp > s.t_max {
        s.t_max = timestamp;
    }
    let req = s.requests;
    for k in 0..count {
        let id = s.interner.intern(s.line_ids[k as usize]);
        let tok = block_tokens(input_length, s.block_size, k, count);
        s.ids.push(id);
        s.toks.push(tok);
        s.total_input_tokens += tok as f64;
    }
    let _ = req;
    s.requests += 1;
    if s.max_events != 0 && s.ids.len() as u32 >= s.max_events {
        s.capped = true;
    }
}

// ---- exported API ----------------------------------------------------------

#[no_mangle]
pub extern "C" fn reset(block_size_override: u32, max_events: u32, warmup_fraction: f64) {
    unsafe {
        STATE = Some(State::new(block_size_override, max_events, warmup_fraction));
    }
}

// Ensure the staging buffer holds at least `len` bytes and return its pointer.
#[no_mangle]
pub extern "C" fn chunk_ptr(len: u32) -> *mut u8 {
    let s = st();
    if (s.staging.len() as u32) < len {
        s.staging.resize(len as usize, 0);
    }
    s.staging.as_mut_ptr()
}

// Process `len` bytes previously written at chunk_ptr(). Returns 1 once the
// event cap is reached so JS can stop streaming.
#[no_mangle]
pub extern "C" fn ingest(len: u32) -> u32 {
    let s = st();
    let len = len as usize;
    let mut start = 0usize;
    let mut idx = 0usize;
    // Work on the staging slice without holding an overlapping borrow of `s`.
    // SAFETY: parse_line only touches the plan/interner fields, never `staging`.
    let staging_ptr = s.staging.as_ptr();
    while idx < len {
        let byte = unsafe { *staging_ptr.add(idx) };
        if byte == b'\n' {
            if s.line.is_empty() {
                let slice: &[u8] = unsafe { std::slice::from_raw_parts(staging_ptr.add(start), idx - start) };
                parse_line(s, slice);
            } else {
                let chunk = unsafe { std::slice::from_raw_parts(staging_ptr.add(start), idx - start) };
                s.line.extend_from_slice(chunk);
                let line = std::mem::take(&mut s.line);
                parse_line(s, &line);
                s.line = line;
                s.line.clear();
            }
            if s.capped {
                return 1;
            }
            start = idx + 1;
        }
        idx += 1;
    }
    if start < len {
        let tail = unsafe { std::slice::from_raw_parts(staging_ptr.add(start), len - start) };
        s.line.extend_from_slice(tail);
    }
    if s.capped {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn finalize() {
    let s = st();
    if !s.line.is_empty() {
        let line = std::mem::take(&mut s.line);
        parse_line(s, &line);
    }
    s.req_start.push(s.ids.len() as u32); // sentinel
    s.unique = s.interner.size;
    // Free the interner before the (memory-heavy) sweep.
    s.interner = Interner::new(0);

    let requests = s.requests;
    let mut warmup = (requests as f64 * s.warmup_fraction).floor();
    if warmup < 0.0 {
        warmup = 0.0;
    }
    let warmup_requests = (warmup as u32).min(requests);
    s.warmup_requests = warmup_requests;
    let warmup_start = s.req_start[warmup_requests as usize];
    s.warmup_event_start = warmup_start;

    let n = s.ids.len();
    let unique = s.unique as usize;
    s.next_use = vec![0u32; n];
    let never = (n + 1) as u32;
    let mut last = vec![EMPTY; unique];
    let mut total_measured = 0.0f64;
    let mut i = n;
    while i > 0 {
        i -= 1;
        let id = s.ids[i] as usize;
        s.next_use[i] = if last[id] != EMPTY { last[id] } else { never };
        last[id] = i as u32;
        if (i as u32) >= warmup_start {
            total_measured += s.toks[i] as f64;
        }
    }
    s.total_measured = total_measured;

    // infinite-cache ceiling (forward seen-set pass)
    let mut seen = vec![false; unique];
    let mut hit = 0.0f64;
    for i in 0..n {
        let id = s.ids[i] as usize;
        if (i as u32) >= warmup_start && seen[id] {
            hit += s.toks[i] as f64;
        }
        seen[id] = true;
    }
    s.ceiling_hit = hit;
}

fn sweep_fifo(s: &State, capacity: u32) -> SimStats {
    let n = s.ids.len();
    if capacity == 0 || n == 0 {
        return SimStats::empty();
    }
    let unique = s.unique as usize;
    let mut in_cache = vec![0u8; unique];
    let never = (n + 1) as u32;
    let mut state_next = vec![never; unique];
    // FIFO holds at most `capacity` live entries and never a stale duplicate
    // (an id is queued only on a miss-with-room and dropped when it reaches the
    // head), so a ring of capacity+1 is exact — and avoids an events-sized queue
    // that would not fit beside a full multi-GB plan in wasm32's 4 GB.
    let qcap = capacity as usize + 1;
    let mut queue = vec![0u32; qcap];
    let mut qhead = 0usize;
    let mut qtail = 0usize;
    let mut size = 0u32;
    let mut stats = SimStats::empty();
    let mut useful_count = 0u32;
    let mut request_cursor = 0usize;
    let ws = s.warmup_event_start;
    for i in 0..n {
        let id = s.ids[i] as usize;
        let is_hit = in_cache[id] == 1;
        if is_hit && (i as u32) >= ws {
            stats.hit += s.toks[i] as f64;
        }
        let nu = s.next_use[i];
        if !is_hit {
            if size >= capacity {
                let victim = queue[qhead] as usize;
                qhead += 1;
                if qhead == qcap {
                    qhead = 0;
                }
                if state_next[victim] < never {
                    useful_count -= 1;
                }
                in_cache[victim] = 0;
                state_next[victim] = never;
                size -= 1;
            }
            in_cache[id] = 1;
            size += 1;
            if nu < never {
                useful_count += 1;
            }
            state_next[id] = nu;
            queue[qtail] = id as u32;
            qtail += 1;
            if qtail == qcap {
                qtail = 0;
            }
        } else {
            if state_next[id] < never {
                useful_count -= 1;
            }
            if nu < never {
                useful_count += 1;
            }
            state_next[id] = nu;
        }
        sample_request(s, &mut request_cursor, i, useful_count, &mut stats);
    }
    stats
}

fn sweep_lru(s: &State, capacity: u32) -> SimStats {
    let n = s.ids.len();
    if capacity == 0 || n == 0 {
        return SimStats::empty();
    }
    let unique = s.unique as usize;
    let mut in_cache = vec![0u8; unique];
    let mut prev = vec![-1i32; unique];
    let mut next = vec![-1i32; unique];
    let never = (n + 1) as u32;
    let mut state_next = vec![never; unique];
    let mut head = -1i32;
    let mut tail = -1i32;
    let mut size = 0u32;
    let mut stats = SimStats::empty();
    let mut useful_count = 0u32;
    let mut request_cursor = 0usize;
    let ws = s.warmup_event_start;
    for i in 0..n {
        let id = s.ids[i] as i32;
        let idu = id as usize;
        let is_hit = in_cache[idu] == 1;
        if is_hit && (i as u32) >= ws {
            stats.hit += s.toks[i] as f64;
        }
        let nu = s.next_use[i];
        if is_hit {
            if state_next[idu] < never {
                useful_count -= 1;
            }
            if nu < never {
                useful_count += 1;
            }
            state_next[idu] = nu;
            if id != tail {
                let p = prev[idu];
                let nx = next[idu];
                if p != -1 {
                    next[p as usize] = nx;
                } else {
                    head = nx;
                }
                if nx != -1 {
                    prev[nx as usize] = p;
                }
                prev[idu] = tail;
                next[idu] = -1;
                if tail != -1 {
                    next[tail as usize] = id;
                }
                tail = id;
            }
        } else {
            if size >= capacity {
                let victim = head;
                let nx = next[victim as usize];
                head = nx;
                if nx != -1 {
                    prev[nx as usize] = -1;
                } else {
                    tail = -1;
                }
                prev[victim as usize] = -1;
                next[victim as usize] = -1;
                if state_next[victim as usize] < never {
                    useful_count -= 1;
                }
                state_next[victim as usize] = never;
                in_cache[victim as usize] = 0;
                size -= 1;
            }
            in_cache[idu] = 1;
            size += 1;
            if nu < never {
                useful_count += 1;
            }
            state_next[idu] = nu;
            prev[idu] = tail;
            next[idu] = -1;
            if tail != -1 {
                next[tail as usize] = id;
            }
            tail = id;
            if head == -1 {
                head = id;
            }
        }
        sample_request(s, &mut request_cursor, i, useful_count, &mut stats);
    }
    stats
}

// Belady/optimal with bypass — exact port of simulateOptimalPlan (typed-array
// cache state + struct-of-arrays max-heap keyed by next-use, lazy versioning).
fn sweep_optimal(s: &State, capacity: u32) -> SimStats {
    let n = s.ids.len();
    if capacity == 0 || n == 0 {
        return SimStats::empty();
    }
    let unique = s.unique as usize;
    let mut present = vec![0u8; unique];
    let mut state_next = vec![0u32; unique];
    let mut state_ver = vec![0u32; unique];
    let mut cache_size = 0u32;
    let mut heap_id: Vec<u32> = Vec::new();
    let mut heap_next: Vec<u32> = Vec::new();
    let mut heap_ver: Vec<u32> = Vec::new();
    let mut stats = SimStats::empty();
    let mut useful_count = 0u32;
    let mut request_cursor = 0usize;
    let ws = s.warmup_event_start;
    let never = (n + 1) as u32;

    macro_rules! heap_push {
        ($id:expr, $nu:expr, $ver:expr) => {{
            let mut idx = heap_id.len();
            heap_id.push($id);
            heap_next.push($nu);
            heap_ver.push($ver);
            while idx > 0 {
                let parent = (idx - 1) >> 1;
                if heap_next[parent] >= heap_next[idx] {
                    break;
                }
                heap_id.swap(parent, idx);
                heap_next.swap(parent, idx);
                heap_ver.swap(parent, idx);
                idx = parent;
            }
        }};
    }

    let mut top_id: u32;
    let mut top_next: u32;
    let mut top_ver: u32;

    for i in 0..n {
        let id = s.ids[i];
        let idu = id as usize;
        let is_hit = present[idu] == 1;
        if is_hit && (i as u32) >= ws {
            stats.hit += s.toks[i] as f64;
        }
        let nu = s.next_use[i];

        let admit = if is_hit {
            true
        } else if cache_size < capacity {
            true
        } else {
            // evictForCandidate(nu)
            let made_room;
            loop {
                if heap_id.is_empty() {
                    made_room = true;
                    break;
                }
                // pop max
                top_id = heap_id[0];
                top_next = heap_next[0];
                top_ver = heap_ver[0];
                let last = heap_id.len() - 1;
                heap_id.swap(0, last);
                heap_next.swap(0, last);
                heap_ver.swap(0, last);
                heap_id.pop();
                heap_next.pop();
                heap_ver.pop();
                let len = heap_id.len();
                if len > 0 {
                    let mut idx = 0usize;
                    loop {
                        let l = idx * 2 + 1;
                        let r = l + 1;
                        let mut largest = idx;
                        if l < len && heap_next[l] > heap_next[largest] {
                            largest = l;
                        }
                        if r < len && heap_next[r] > heap_next[largest] {
                            largest = r;
                        }
                        if largest == idx {
                            break;
                        }
                        heap_id.swap(largest, idx);
                        heap_next.swap(largest, idx);
                        heap_ver.swap(largest, idx);
                        idx = largest;
                    }
                }
                let tu = top_id as usize;
                if present[tu] == 0 || state_ver[tu] != top_ver || state_next[tu] != top_next {
                    continue; // stale heap entry
                }
                if top_next > nu {
                    if state_next[tu] < never {
                        useful_count -= 1;
                    }
                    present[tu] = 0;
                    cache_size -= 1;
                    made_room = true;
                    break;
                }
                heap_push!(top_id, top_next, top_ver);
                made_room = false;
                break;
            }
            made_room
        };

        if admit {
            // pushState(id, nu)
            let version = if present[idu] == 1 { state_ver[idu] + 1 } else { 1 };
            if present[idu] == 0 {
                present[idu] = 1;
                cache_size += 1;
            } else if state_next[idu] < never {
                useful_count -= 1;
            }
            if nu < never {
                useful_count += 1;
            }
            state_next[idu] = nu;
            state_ver[idu] = version;
            heap_push!(id, nu, version);
        }
        sample_request(s, &mut request_cursor, i, useful_count, &mut stats);
        // Bound the lazily-versioned heap: when stale entries pile up, compact to
        // the live cache contents and re-heapify (Floyd). For upload traces (all
        // input events) the only next-use ties are among never-again-used blocks,
        // whose eviction order can change no future hit — so this is result-
        // identical while keeping the heap ~cache_size instead of O(events),
        // which is what lets a full multi-GB trace's Optimal sweep fit in 4 GB.
        if heap_id.len() > (cache_size as usize) * 2 + 4096 {
            let mut w = 0usize;
            let hl = heap_id.len();
            for r in 0..hl {
                let id2 = heap_id[r] as usize;
                if present[id2] == 1 && state_ver[id2] == heap_ver[r] && state_next[id2] == heap_next[r] {
                    heap_id[w] = heap_id[r];
                    heap_next[w] = heap_next[r];
                    heap_ver[w] = heap_ver[r];
                    w += 1;
                }
            }
            heap_id.truncate(w);
            heap_next.truncate(w);
            heap_ver.truncate(w);
            for i2 in (0..w / 2).rev() {
                let mut idx = i2;
                loop {
                    let l = idx * 2 + 1;
                    let r = l + 1;
                    let mut largest = idx;
                    if l < w && heap_next[l] > heap_next[largest] {
                        largest = l;
                    }
                    if r < w && heap_next[r] > heap_next[largest] {
                        largest = r;
                    }
                    if largest == idx {
                        break;
                    }
                    heap_id.swap(largest, idx);
                    heap_next.swap(largest, idx);
                    heap_ver.swap(largest, idx);
                    idx = largest;
                }
            }
        }
    }
    stats
}

#[no_mangle]
pub extern "C" fn sweep(cache_blocks: u32, policy: u32) -> f64 {
    run_sweep(cache_blocks, policy).hit
}

fn run_sweep(cache_blocks: u32, policy: u32) -> SimStats {
    let s = st();
    if cache_blocks == 0 || s.ids.is_empty() {
        return SimStats::empty();
    }
    let effective_cache_blocks = if s.unique > 0 && cache_blocks >= s.unique { s.unique } else { cache_blocks };
    match policy {
        POLICY_FIFO => sweep_fifo(s, effective_cache_blocks),
        POLICY_LRU => sweep_lru(s, effective_cache_blocks),
        POLICY_OPTIMAL => sweep_optimal(s, effective_cache_blocks),
        _ => SimStats::empty(),
    }
}

#[no_mangle]
pub extern "C" fn sweep_stats(cache_blocks: u32, policy: u32) -> u32 {
    let stats = run_sweep(cache_blocks, policy);
    let s = st();
    s.stats_out.clear();
    s.stats_out.push(stats.hit);
    s.stats_out.push(stats.useful_sum);
    s.stats_out.push(stats.samples);
    s.stats_out.as_ptr() as u32
}

// Temporal statistics, mirroring computeTimeSeries() in kv-cache-lab.js: walk
// requests in timestamp order, bucket arriving/reused tokens into 48 wall-clock
// bins, and histogram the gap since each block's previous use. Results are packed
// into a single f64 buffer; JS reads it and hands the accumulators to the shared
// buildTimeSeriesResult() so the panel matches the JS path exactly.
const TIME_BUCKETS: usize = 48;
// REUSE_GAP_BINS upper bounds (seconds), matching the JS bins.
const GAP_MAX: [f64; 10] = [1.0, 3.0, 10.0, 30.0, 60.0, 180.0, 600.0, 1800.0, 3600.0, f64::INFINITY];

#[no_mangle]
pub extern "C" fn compute_time_series() -> u32 {
    let s = st();
    let requests = s.requests as usize;
    if requests < 2 {
        return 0;
    }
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for r in 0..requests {
        let t = s.ts[r];
        if t < t_min {
            t_min = t;
        }
        if t > t_max {
            t_max = t;
        }
    }
    if !(t_max > t_min) {
        return 0; // no usable timestamp spread -> JS shows no temporal panel
    }
    let mut order: Vec<u32> = (0..requests as u32).collect();
    order.sort_by(|&a, &b| s.ts[a as usize].partial_cmp(&s.ts[b as usize]).unwrap_or(std::cmp::Ordering::Equal));

    let span = t_max - t_min;
    let dt = span / TIME_BUCKETS as f64;
    let mut bucket_total = [0f64; TIME_BUCKETS];
    let mut bucket_hit = [0f64; TIME_BUCKETS];
    let mut gap_tokens = [0f64; 10];
    let mut gap_count = [0f64; 10];
    let unique = s.unique as usize;
    let mut last_ts = vec![0f64; unique];
    let mut has_last = vec![0u8; unique];
    let mut total_tokens = 0f64;
    let mut reuse_tokens = 0f64;

    for &r in order.iter() {
        let r = r as usize;
        let t = s.ts[r];
        let mut bucket = ((t - t_min) / span * TIME_BUCKETS as f64).floor() as i64;
        if bucket >= TIME_BUCKETS as i64 {
            bucket = TIME_BUCKETS as i64 - 1;
        }
        if bucket < 0 {
            bucket = 0;
        }
        let bucket = bucket as usize;
        let start = s.req_start[r] as usize;
        let end = s.req_start[r + 1] as usize;
        for k in start..end {
            let id = s.ids[k] as usize;
            let tok = s.toks[k] as f64;
            bucket_total[bucket] += tok;
            total_tokens += tok;
            if has_last[id] == 1 {
                bucket_hit[bucket] += tok;
                reuse_tokens += tok;
                let gap = t - last_ts[id];
                let mut bin = 0usize;
                while bin < 9 && gap >= GAP_MAX[bin] {
                    bin += 1;
                }
                gap_tokens[bin] += tok;
                gap_count[bin] += 1.0;
            }
            last_ts[id] = t;
            has_last[id] = 1;
        }
    }

    let mut out: Vec<f64> = Vec::with_capacity(6 + 2 * TIME_BUCKETS + 20);
    out.push(t_min);
    out.push(t_max);
    out.push(span);
    out.push(dt);
    out.push(total_tokens);
    out.push(reuse_tokens);
    out.extend_from_slice(&bucket_total);
    out.extend_from_slice(&bucket_hit);
    out.extend_from_slice(&gap_tokens);
    out.extend_from_slice(&gap_count);
    s.time_out = out;
    s.time_out.as_ptr() as u32
}

#[no_mangle]
pub extern "C" fn time_series_len() -> u32 {
    st().time_out.len() as u32
}

// ---- summary getters -------------------------------------------------------

#[no_mangle]
pub extern "C" fn requests() -> u32 {
    st().requests
}
#[no_mangle]
pub extern "C" fn events() -> u32 {
    st().ids.len() as u32
}
#[no_mangle]
pub extern "C" fn unique_blocks() -> u32 {
    st().unique
}
#[no_mangle]
pub extern "C" fn parse_errors() -> u32 {
    st().parse_errors
}
#[no_mangle]
pub extern "C" fn skipped() -> u32 {
    st().skipped
}
#[no_mangle]
pub extern "C" fn missing_block_size() -> u32 {
    st().missing_block_size
}
#[no_mangle]
pub extern "C" fn inconsistent_block_size() -> u32 {
    st().inconsistent_block_size
}
#[no_mangle]
pub extern "C" fn was_capped() -> u32 {
    if st().capped {
        1
    } else {
        0
    }
}
#[no_mangle]
pub extern "C" fn block_size() -> u32 {
    st().block_size
}
#[no_mangle]
pub extern "C" fn warmup_requests() -> u32 {
    st().warmup_requests
}
#[no_mangle]
pub extern "C" fn total_input_tokens() -> f64 {
    st().total_input_tokens
}
#[no_mangle]
pub extern "C" fn total_measured() -> f64 {
    st().total_measured
}
#[no_mangle]
pub extern "C" fn ceiling_hit() -> f64 {
    st().ceiling_hit
}
#[no_mangle]
pub extern "C" fn t_min() -> f64 {
    let s = st();
    if s.t_max > s.t_min {
        s.t_min
    } else {
        0.0
    }
}
#[no_mangle]
pub extern "C" fn t_max() -> f64 {
    let s = st();
    if s.t_max > s.t_min {
        s.t_max
    } else {
        0.0
    }
}
