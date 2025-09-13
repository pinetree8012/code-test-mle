## Part 1

### Goal

The objective is to build a fraud detection feature that measures transaction velocity. Specifically, for each transaction, we count how many transactions from the same `paymentMethodIssuer` occurred in the preceding 24 hours. This allows us to detect sudden spikes in activity from a given issuer.  

Each output row includes:
- `transactionId`
- `transactionTime`
- `paymentMethodIssuer`
- `fraudulent`
- `issuer_velocity_24h` : the count of issuer-specific transactions within the last 24 hours (rolling window aligned to the transaction timestamp)
- (Optional: customer-level information, if required)

---

### Approach

1. **Preprocessing**  

   Flatten the nested dataset into a transaction-level table, since the feature should be computed per transaction.  
   Each customer may have multiple orders, payment methods, and transactions:  

   ```json
   {
      "fraudulent": true,
      "customer": { ... },
      "orders": [ ... ],
      "paymentMethods": [ ... ],
      "transactions": [ ... ]
   }
   ```
   → Build a mapping from paymentMethodId → paymentMethodIssuer

   → Extract all transactions into a flat table with {transactionId, timestamp, paymentMethodIssuer}.

2. **Sorting**

   Sort all transactions chronologically to enable efficient sequential processing.

3. **Sliding Window Counting**

   Use a two-pointer sliding window to compute the feature.

   As the right pointer advances, the left pointer shifts forward whenever transactions fall outside the 24-hour window.

   The window size (right - left) gives the count of prior transactions for that issuer.

   **Complexity Analysis**
   - Sorting: O(nlogn)
   - Window scan: O(n)
   - Overall: O(nlogn) (dominated by sorting)
   - Memory: O(window size per issuer)

   **Advantages**
   - Efficient and scalable
   - Adaptable to streaming pipelines

4. **Future Work**

   For large scale production, I plan to extend this design with `Ray` to parallelise the sliding-window computation across nodes. This will allow the python-based logic to scale horizontally while retaining the simplicity of the two-pointer method.


## Part 2: PyTorch Model for Inference

### Goal

Build a simple fraud-prevention inference service using a PyTorch model.

---

### Approach

1. **Mean&Std**

    To ensure consistent preprocessing, I logged the mean and standard deviation values when running create_model.py, and these figures are reused during inference.

2. **TorchScript Integration**

    The model was serialised into TorchScript (`scripted_model = torch.jit.script(model)`). which is a more efficient, deployable form.

3. **Inference Workflow**

    - Load: torch.jit.load() retrieves the serialised TorchScript model.
    - Evaluation Mode: model.eval() disables training-specific behaviour such as Dropout or BatchNorm updates.
    - Inference Mode: torch.inference_mode() further improves performance by disabling gradient tracking and metadata creation. This is more efficient than torch.no_grad().

4. **Deployment Considerations**

    - Warm-up Runs: Supplying dummy inputs at startup allows the JIT graph to optimise and reduces first-call latency.
    - Hardware Choice: For larger models, careful consideration of CPU versus GPU inference is required.
    - Batching: Under heavier traffic, batching strategies can significantly improve throughput, particularly when inputs follow a JSON structure similar to that in `customers.jsonl`, as they can then be processed at once.

5. **Key Metrics**

    To evaluate deployment quality, the following metrics can be considered:

    - Traffic: Requests per second (RPS/QPS), hourly and daily totals, and peak load.
    - Latency: End-to-end request times (p50/p95/p99), including cold-start latency.
    - Saturation: CPU/GPU utilisation, memory/VRAM consumption, queue length, and concurrency.

### Performance snapshot

    - Model load: 0.022s
    - Cold start (1st inference): 0.067s
    - Warm-up: 10 calls (fixed input shape): avg 0.003s
