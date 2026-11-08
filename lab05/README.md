# Late Deliveries
Predicts whether a QuickBite food delivery order will arrive late at the moment it is placed.

## Decisions
- Features: distance_km, items, hour, weather, day_type, vehicle, courier_months.
- Excluded: prep_minutes, customer_rating, order_id to prevent data leakage.
- Model: LogisticRegression (balanced class weight) chosen via 5-fold CV.
- Threshold: Optimal threshold of 0.35 chosen to minimize cost.

## Results
- Model policy cost: $2,308.00 (beats Flag Nobody at $2,640 and Flag Everybody at $3,000).
