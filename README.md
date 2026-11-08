```
INSTAMART CUSTOMER INTELLIGENCE AND SUPPORT COPILOT
===================================================
All data is made up (synthetic). All money is in INR.
Python 3.14 was used (the brief needs 3.11 or newer).


HOW TO REPRODUCE
----------------
1. python -m venv .venv  (then activate it: .venv\Scripts\activate)
2. pip install -r requirements.txt
3. python generate_data.py
4. python src/clean.py
5. python src/validate.py         -> prints ALL CHECKPOINTS PASSED
6. python src/analysis_a.py
7. python src/segments.py
8. python src/lapse_model.py
9. python src/triage.py
10. python src/rag.py
11. python src/run_agent_demo.py
12. python src/check_layout.py    -> prints LAYOUT OK

Run them in this order. Part B needs the clean data, Part C needs
outputs/customer_segments.csv and Part D needs data/instamart.db.
generate_data.py and mock_llm.py are the given files, not changed.


=====================================================
PART A - CLEAN THE DATA
=====================================================

AUDIT OF THE RAW DATA
Shapes: orders 13084 x 9, customers 1200 x 5, products 40 x 4.
Missing values: delivery_minutes 2541 (19.42%), order_amount 53 (0.41%),
age 61 (5.08%), gender 30 (2.50%). Every other column has none.

  raw order rows ............................ 13,084
  exact duplicate rows ...................... 119
  rows with status pending .................. 656
  distinct order_id ......................... 12,309
  order_amount that is not a number ......... 120
  negative delivery_minutes ................. 106
  blank delivery_minutes .................... 2,541
  blank age ................................. 61
  age outside 10-100 ........................ 6
  distinct raw city spellings ............... 24
  distinct non-blank gender spellings ....... 10
  blank gender .............................. 30

CLEANING RULES (in this order) AND HOW MANY THEY CHANGED
Customers:
  - city: strip spaces + title case ......... 476 values, 24 spellings -> 6 cities
  - gender -> male / female / unknown ....... 1170 mapped, 30 blanks -> unknown
  - age outside 10-100 -> missing ........... 6
  - missing age -> median age of the city ... 67 (61 blank + 6 impossible)
  - signup_date -> datetime ................. 1200
Orders:
  - order_ts -> datetime .................... 13084
  - exact duplicate rows removed ............ 119
  - keep latest order_ts per order_id ....... 656 removed
  - bad amounts rebuilt from price .......... 112
  - negative delivery_minutes -> missing .... 105
  - completed orders: fill delivery time
    with median of the product category ..... 774
  - cancelled orders: left missing .......... 1199

 105 and 112 are here, but 106 and 120 in the audit because a few bad values were
inside rows that an earlier rule had already removed.

Result: 12,309 orders (11,110 completed, 1,199 cancelled), no pending rows,
completed revenue INR 2,998,668.80. validate.py checks 14 numbers and
prints ALL CHECKPOINTS PASSED.

SQL RESULTS (queries are in sql/queries.sql)
  Q1  30 customers
  Q2  30 rows, total INR 2,998,668.80, biggest = Bengaluru / Electronics
      INR 233,909.45
  Q3  11 products
  Q4  25 customers
  Q5  550 customers above the average of INR 2,552.06
  Q6  top customer per city: Bengaluru C0082 12,856.20, Chennai C0930
      10,151.40, Delhi C0291 13,742.60, Hyderabad C0743 12,730.05,
      Mumbai C0693 14,766.55, Pune C0426 10,377.10.
      8 rows have RANK different from ROW_NUMBER.
  Q7  183 daily rows, last cumulative revenue INR 2,998,668.80

Ties make RANK and ROW_NUMBER differ: when two customers in a city have
the same total, RANK gives them the same number but ROW_NUMBER still gives
each one a different number.

PANDAS CHECK
The pivot table (margins=True) grand total is INR 2,998,668.80, the same as
the SQL total.
  Bengaluru: sum 523,643.40, mean 276.47, count 1,894
  Mumbai:    sum 479,058.50, mean 297.00, count 1,613
order_amount (completed): mean 269.91, median 135.00, skewness 4.61,
Q1 64.80, Q3 297.00, upper fence 645.30, 897 outliers above, 0 below.

ABOUT THE DATA (my answers)
Q1: The data is about the details of each orders like order amount,quantity,which customer orederd (order id)

 Q2: The data looks a little bit right-skewed

 Q3: There are 897 outliers above the upper fence but none below the lower fence

 Q4: There is a strong negative correlation between order_amount and discount_pct and a positive correlation between order_amount and quantity


=====================================================
PART B - SEGMENTS AND LAPSE MODEL
=====================================================

RFM (snapshot 2025-10-01, completed orders only)
1,175 customers. Recency median 19, min 0, max 180. Frequency mean 9.46,
max 33. Monetary mean 2,552.06, max 14,766.55.

K-MEANS (scaled with StandardScaler, n_init=10, random_state=42)
  K    inertia   silhouette
  2    1820.8    0.465
  3    1268.2    0.445
  4     743.1    0.526
  5     617.4    0.480
  6     503.2    0.482
  7     408.5    0.416
  8     366.5    0.412

I chose K = 4. The inertia drops a lot from K=3 to K=4 (1268 -> 743) and
only a little after that (743 -> 617), so the elbow is at 4. K=4 also has
the best silhouette (0.526), and four groups are easy for the business to use.
Inertia at K=8 is always lower than at K=2 because more centres put every
customer closer to a centre (with one cluster per customer it would be 0), so
lower inertia alone cannot pick K. We also need the elbow, the silhouette and
whether the groups make sense.


PCA on the scaled RFM: explained variance 0.634 and 0.218, total 0.852.
Saved: outputs/customer_segments.csv

LAPSE PROBLEM
Cutoff 2025-09-01. Cohort: customers with at least one completed order before
the cutoff = 1,156. Lapsed = no completed order in September = 443 (38.32%).
Split 80/20 with stratify and random_state=42: 924 train, 232 test.
Baseline (always "not lapsed"): accuracy 61.6%, lapsed F1 0.
Features (all from orders before the cutoff): recency, frequency, monetary,
average discount, cancellation rate, average delivery minutes, favourite
category, city, age, gender, tenure in days.

The scaler learned a frequency mean of 8.115 from the training rows only,
and the full cohort mean is 8.029. They differ because the training rows are
only part of the cohort (the test rows are left out).
No feature column has a missing value (checked in code).

TUNING (GridSearchCV, cv=5, scoring f1, preprocessing inside a Pipeline)
  Logistic regression: best C = 100, CV F1 0.644
  Random forest: best max_depth 8, min_samples_leaf 10, CV F1 0.683

TEST SET (lapsed class)
  model                precision  recall  F1     confusion [[TN FP] [FN TP]]
  Logistic regression  0.615      0.629   0.622  [[108 35] [33 56]]
  Random forest        0.648      0.663   0.656  [[111 32] [30 59]]

I picked the random forest. Its test F1 is 0.656 against 0.622, so it is
better on both precision and recall, and it is above the 0.55 needed.
Missing a customer who is about to lapse means losing their future orders,
while wrongly targeting a loyal customer only costs a small offer. So catching
more lapsers matters most, and the forest catches more of them.

CLASS-BALANCE EXPERIMENT (only the training set was over-sampled: 570 / 570)
  logistic regression: recall 0.629 -> 0.865, precision 0.615 -> 0.592,
  F1 0.622 -> 0.703.
Recall moved the most. The team would catch many more lapsing customers but
would also contact more loyal customers by mistake, which is fine if the
offer is cheap.

IMPORTANT FEATURES
Top 5 in the forest: frequency 0.284, monetary 0.275, recency 0.123,
average delivery 0.062, cancel rate 0.048.
  - frequency: customers who order often are less likely to lapse
  - monetary: big spenders matter, but it overlaps with frequency
  - recency: a long time since the last order means more risk
  - average delivery: slow deliveries may push customers away
  - cancel rate: more cancelled orders means more risk
Logistic regression signs: frequency negative (-1.627), recency positive
(+0.042), monetary positive (+0.222). The positive monetary sign is odd
because it overlaps with frequency, so I do not read it alone.

SAVED MODEL
outputs/lapse_model.joblib was saved with joblib.dump, loaded again with
joblib.load, and gave exactly the same test predictions.
outputs/retention_list.csv has the 10 Premium test customers most likely to
lapse. 0 of them truly lapsed. This list is illustrative and not a
leakage-free evaluation, because the Premium flag uses the whole window.

LEAKAGE STATEMENTS
  - No encoder, scaler or resampler saw the test rows.
  - No data from 2025-09-01 onwards was used in the features.
  - The segment labels from the K-means step were not used as features.


=====================================================
PART C - SUPPORT TICKETS
=====================================================
164 tickets. 4 are empty (skipped_empty) and 4 are too long at 734 estimated
tokens (flagged_long). Neither type is ever sent to the model.

REDACTION: 55 emails, 91 phone numbers, 132 order ids = 278 in total.
Running the three patterns again on the redacted text finds 0. An assert in
the code checks that no prompt sent to the model contains any of them.

RETRIES: 19 temporary errors in a full run, all fixed by a retry. Every wait
was between 0.045 and 0.0575 seconds. No ticket ended with an error.

STATUSES in one full run: 135 classified_json, 12 classified_keyword,
9 flagged_invalid_output.

ZERO-SHOT VS FEW-SHOT
  prompt     classified  accuracy        total tokens  tokens per ticket
  zero-shot  147         85.7% (126/147) 8,152         55.5
  few-shot   147         100% (147/147)  19,540        132.9

Zero-shot confusion table (all 21 mistakes):
  fraud predicted as billing ........ 11
  technical predicted as billing .... 10
  (everything else was correct)

A fraud ticket sent to billing waits instead of going to the security team, so I think
the extra tokens are worth it when a wrong route is expensive. If the tickets
were easy, zero-shot would be enough.

BUDGET DEMO: TOKEN_BUDGET = 6000 with the zero-shot prompt left 41 tickets as
budget_exhausted (saved in outputs/triage_budget_demo.csv).

FINAL OUTPUT: outputs/triage_results.csv (few-shot, budget 100000) has 164
rows. Routes: support_queue 102, security_team 34, human_review 17,
premium_desk 11.
Checked by rules: the 147 classified tickets (135 json + 12 keyword).
Sent to humans: 17 tickets (9 invalid output, 4 empty, 4 too long).


=====================================================
PART D - POLICY ASSISTANT AND REFUND AGENT
=====================================================
CHUNKS: 15 in total (refund 4, delivery 3, escalation 3, payment 3, privacy 2).

TF-IDF: TF (term frequency) says how often a word appears in a chunk. IDF
(inverse document frequency) says how rare a word is across all chunks, so
rare words count more. A limit of TF-IDF is that it only matches words, so two
phrasings with the same meaning but different words (like "money back" and
"refund") do not match well.

EVALUATION (14 questions)
  overlap 10: expected_doc in top 3 = 12/12, expected_fact in answer = 12/12,
              fallback answers = 2/2
  overlap 0 (13 chunks): 12/12, 10/12, 2/2
With overlap 0 two questions broke: "What delivery time makes a completed
order eligible for a full refund?" and "How soon is money returned when a
payment fails but is deducted?". The right document was found, but the
answer sentence was most likely cut across a chunk boundary, so the fact was
missing from the answer.

AGENT RESULTS (7 scenarios)
  scenario  status                   refund INR  model calls  tokens
  S1        REFUND                   128.25      5            912
  S2        REFUND                   135.00      5            912
  S3        ESCALATE                 1999.00     5            899
  S4        DENIED                   0           4            617
  S5        DENIED                   0           3            346
  S6        NOT_FOUND                0           2            78
  S7        FALLBACK_MAX_ITERATIONS  0           6            3018

SAFETY DEMOS (S1)
  - token_limit=200 ended with FALLBACK_TOKEN_LIMIT
  - time_limit_seconds=0 ended with FALLBACK_TIMEOUT

CRASH AND RESUME (S1): the agent crashed after the second tool. Tools run
across the crash and the resume, in order: get_order, search_policy,
calculate_refund, issue_refund. That is 4 runs, none repeated. The final
result was REFUND 128.25.

REACT VS PLAN-AND-EXECUTE
My ReAct agent asks the model one step at a time. It used 5 model calls for
S1, 3 for S5 and 2 for S6 (912, 346 and 78 tokens). It sends the whole history
again at every call, so tokens grow with each step (S7: 3,018 tokens for 6
calls). Plan-and-execute asks for the whole list of steps once, so it would
likely use fewer model calls and fewer tokens. But when something unexpected
comes back, like NOT_FOUND (S6) or a cancelled order (S5), ReAct reads the
result and stops with the right answer straight away. Plan-and-execute would
keep following its plan, for example calculating a refund for an order that
does not exist, until an error forces a new plan and another model call. For
this refund workflow I would choose ReAct. The next step depends on what
get_order returns, the runs are short (2 to 5 calls), and money is involved,
so safety matters more than saving a few tokens. I would use
plan-and-execute only for long workflows with fixed steps.


