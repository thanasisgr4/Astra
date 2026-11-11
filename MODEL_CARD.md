# ASTRAIA 5.0.1 — Model card

Research prototype for fixed-policy replenishment of one depot. Not flight software, a Starship simulator, operational assurance or a certified reliability model. No dependency on Gateway or L1.

## Boundary
Elapsed days; kg of one aggregate compatible commodity; nominal USD. No fuel/oxidizer mixture tracking, orbit propagation, delta-v, electric propulsion solver, docking compatibility, thermal solver, reusable fleet lifecycle, or downstream transport. External lead times and delivery-service prices must be supplied and justified by the user. Tanker propulsion is outside the commodity mass ledger and must be covered by the external service price where appropriate. No total initial-LEO-mass claim.

## Dispatch and unloading
Policy: fixed departure times, identical load per trip, retained inventory floor. Arrival = departure + external lead + campaign-common delay + individual trip delay. One non-preemptive unloading server, arrival-order priority, trip-index tie break. Completion = max(arrival, previous completion) + loaded mass / unloading rate. Duration deliberately uses loaded mass before losses. Supply becomes available only at completion, not gradually during unloading.

Transit retention = load * (1-daily transit loss fraction) ** elapsed days from departure through unloading completion, including waiting. Offered supply = retained mass * transfer efficiency. This assumed loss law is not a cryogenic thermal calculation.

## Depot and demand
Storage retention follows (1-daily storage loss fraction)**elapsed days. Supply beyond capacity is rejected/overflow. At each timestamp: decay stock; book all deposits; release all demands; allocate earliest-deadline-first; expire unmet demands and retry after a blocked head expires. No bypass of a blocked head. Whole-request, instantaneous depot handover; no consumer storage or finite outbound transfer modeled. Delivery exactly at the deadline is allowed. Retained floor remains at the depot; it is not consumer reserve fuel.

Campaign service means ALL requests receive their full mass within their windows. Missed service does not imply spacecraft destruction, cargo loss or danger to people.

Depot ledger: initial stock + offered completed supply = demand handovers + storage losses + overflow + final stock. Offered supply includes overflow. Separate trip ledger: load = transit loss + transfer loss + offered mass. No double-counting between ledgers. Trips finishing after the horizon contribute no depot supply but are charged. Their eventual offered mass in the audit is a projection, not stock remaining in transit at the horizon; no full-system in-transit mass balance is claimed.

## Cost
Attempted cost = [fixed campaign charge + committed trip count * (delivery-service charge per trip + load * commodity price)] * common sampled cost multiplier. All scheduled trips are charged, even if late or unsuccessful. No revenue, refund, salvage, discounted cash flow, failure penalty, infrastructure construction calculation, fleet reuse amortisation, or cost-per-success inference. Initial stock acquisition and existing depot assets are outside this boundary; include an appropriate external campaign allocation if needed. Avoid double-counting launch cost already included in service quotes.

## Uncertainty and comparison
Individual delays: Bernoulli occurrence then uniform magnitude. Common campaign delay: separate Bernoulli occurrence then uniform magnitude added to all trips. Cost multiplier uniform within declared relative half-range. The examples are entirely synthetic and uncalibrated. Shared samples across policies correspond to trip index, not actual shared flight identifiers. Independence assumptions hold only as declared; no hardware failures are sampled.

Candidate grid: 2/3/4 trips; 4/8 day cadence; first departure 0/3; retained floor 0/500 kg. Invalid candidates omitted. Custom policy optional. This is a finite illustrative search, not a globally optimal solution.

Select the least mean-cost candidate whose selection-sample two-sided 95% Wilson lower bound reaches the declared service target. Then test only the selected policy on independent model draws, seed + 1000003. Explicitly report holdout target pass/fail; do not recommend if no selection candidate qualifies. Repeated tuning after seeing holdout requires a fresh predeclared confirmation case. Wilson intervals quantify sampling uncertainty conditional on the model, not physical-model validity.

Pareto uses estimated service and mean attempted cost, not statistically proven dominance. Upper-5% cost average uses the largest ceil(0.05*N) sampled costs, not mission-loss severity. Stress cases are capacity -20%, lead +3 days, storage loss x2, one at a time; redesign costs are not modeled. They are diagnostics, not free engineering improvements.

## Evidence
Each scenario field requires declared provenance. 'Published' and 'measured' are operator labels, NOT automatically verified facts. Evidence export includes inputs, policies, seeds and scenario-value SHA-256. The hash is an identifier, not certification or a digital signature. Both example datasets are synthetic. Conceptual relevance of propellant logistics does not validate their numerical parameters.

## Deployment and verification status
The 5.0.1 delivery contains a matched sibling-module loader and explicit version check to help diagnose missing/wrong engine.py. It is a deployment repair set, not a merger with the unavailable v4.1 source. Automated core tests and source syntax checks are recorded in TEST_RESULTS.txt. Streamlit is unavailable in the build environment; browser behavior and Cloud deployment are NOT executed or verified here. No real-mission calibration, independent technical review, ESA endorsement or acceptance guarantee. The previous 5.0 build test report must not substitute for this delivery's test report.
