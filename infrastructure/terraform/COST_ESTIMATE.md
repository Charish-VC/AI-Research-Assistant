# Cost Estimate — AI Research Intelligence Platform

Monthly cost estimate assuming **light usage** (~1,000 documents/month, us-east-1).

## Summary

| Service    | Estimated Monthly Cost |
|------------|----------------------:|
| S3         |               < $0.10 |
| DynamoDB   |               < $0.10 |
| SQS        |               < $0.01 |
| Lambda     |               < $0.05 |
| CloudWatch |               < $0.50 |
| **Total**  |          **< $0.76**  |

> [!NOTE]
> AWS Free Tier may cover most of these costs for the first 12 months.

---

## Detailed Breakdown

### S3 ($0.023/GB-month storage, $0.005/1K PUT, $0.0004/1K GET)

| Item              | Quantity          | Cost      |
|-------------------|-------------------|-----------|
| Storage           | ~500 MB           | $0.012    |
| PUT requests      | 1,000/month       | $0.005    |
| GET requests      | 5,000/month       | $0.002    |
| **S3 Subtotal**   |                   | **~$0.02**|

### DynamoDB (PAY_PER_REQUEST)

| Item              | Quantity          | Cost      |
|-------------------|-------------------|-----------|
| Write requests    | 1,000/month       | $0.00125  |
| Read requests     | 5,000/month       | $0.00125  |
| Storage           | < 1 GB            | $0.25     |
| PITR backup       | < 1 GB            | $0.20     |
| **DynamoDB Sub.** |                   | **~$0.05**|

### SQS ($0.40/million requests after free tier)

| Item              | Quantity          | Cost      |
|-------------------|-------------------|-----------|
| Requests          | ~5,000/month      | Free tier |
| **SQS Subtotal**  |                   | **~$0.00**|

### Lambda ($0.20/million requests, $0.0000166667/GB-s)

| Item              | Quantity                     | Cost       |
|-------------------|------------------------------|------------|
| Invocations       | 1,000/month                  | Free tier  |
| Compute (512 MB)  | 1,000 × ~10s avg = 5,000 GB-s| ~$0.04     |
| **Lambda Sub.**   |                              | **~$0.04** |

### CloudWatch ($0.50/metric, $0.10/alarm, $0.50/GB ingested)

| Item              | Quantity          | Cost        |
|-------------------|-------------------|-------------|
| Log storage       | ~100 MB           | $0.05       |
| Log ingestion     | ~100 MB           | $0.05       |
| Metric alarms (2) | 2 standard alarms | $0.20       |
| **CW Subtotal**   |                   | **~$0.30**  |

---

## Notes

- All prices are us-east-1 as of March 2026.
- DynamoDB on-demand pricing: $1.25/million WCU, $0.25/million RCU.
- Free Tier includes 1M Lambda requests, 1M SQS requests, 5 GB S3 storage, and 25 GB DynamoDB storage per month.
- Bedrock costs are **not included** — they depend on model selection and token volume.
- Data transfer costs are negligible at this scale and omitted.
