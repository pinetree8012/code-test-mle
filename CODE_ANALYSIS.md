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

   For large scale production, I plan to extend this design with Ray to parallelize the sliding-window computation across nodes. This will allow the python-based logic to scale horizontally while retaining the simplicity of the two-pointer method.
