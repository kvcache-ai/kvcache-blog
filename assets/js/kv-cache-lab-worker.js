(function (root) {
  "use strict";

  function labModule() {
    if (root.KVCacheLab) return root.KVCacheLab;
    if (typeof require === "function") return require("./kv-cache-lab.js");
    throw new Error("KVCacheLab is not loaded");
  }

  function ensureBrowserScripts(message) {
    if (root.KVCacheLab) return;
    if (typeof importScripts !== "function") return;
    if (!message.calculatorScriptUrl || !message.labScriptUrl) {
      throw new Error("Worker script URLs are missing");
    }
    importScripts(message.calculatorScriptUrl, message.labScriptUrl);
  }

  function runWorkerJob(message) {
    ensureBrowserScripts(message || {});
    const lab = labModule();
    const trace = lab.generateTrace(message.preset, message.params, message.seed);
    const sweep = lab.sweepCapacity(trace, message.model, message.settings || {});
    return {
      jobId: message.jobId,
      cacheKey: message.cacheKey,
      preset: message.preset,
      trace,
      sweep,
    };
  }

  function handleMessage(message, postResult) {
    if (!message || message.type !== "run") return;
    try {
      postResult({ type: "result", jobId: message.jobId, result: runWorkerJob(message) });
    } catch (error) {
      postResult({ type: "error", jobId: message.jobId, error: error.message || String(error) });
    }
  }

  if (typeof self !== "undefined" && typeof self.addEventListener === "function") {
    self.addEventListener("message", (event) => {
      handleMessage(event.data, (payload) => self.postMessage(payload));
    });
  }

  if (typeof module === "object" && module.exports) {
    module.exports = { runWorkerJob };
    try {
      const workerThreads = require("node:worker_threads");
      if (!workerThreads.isMainThread && workerThreads.parentPort) {
        workerThreads.parentPort.on("message", (message) => {
          handleMessage(message, (payload) => workerThreads.parentPort.postMessage(payload));
        });
      }
    } catch (error) {
      // Browser builds and plain CommonJS imports do not need worker_threads.
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
