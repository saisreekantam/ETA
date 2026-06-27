# Business Impact

Every figure below is either sourced (cited inline) or explicitly marked as our own assumption.
Do not present an assumption as a sourced fact on stage — say "we estimate" out loud.

## The cost problem, in rupees

- **6,500+ fatal workplace accidents/year** in India's heavy industrial sector (FY2023,
  DGFASLI) — this excludes most of mining and construction, so the true figure is higher.
  This is the same statistic the problem statement opens with.
- Separately, DGFASLI's narrower formal-factory-sector data (2017-2020) shows **~3 deaths
  and 11 injuries every day** among *registered factory* workers specifically — a different,
  smaller slice than the 6,500 figure above, cited here so we don't conflate the two scopes
  if asked.
- **Compensation per fatality is not the trivial ₹1.2 lakh statutory floor it's sometimes
  quoted as.** Under the Employees' Compensation Act, actual payouts are 50% of monthly
  wage × an age-based multiplier, or ₹1,20,000, whichever is higher. Worked examples at
  real wage levels: a 30-year-old on ₹15,000/month → **₹15,59,850**; a 32-year-old on
  ₹25,000/month → **₹25,53,125** (figures from published compensation-calculator worked
  examples, India Briefing / BimaKavach). This is direct statutory compensation only — it
  excludes legal liability, production downtime, regulatory fines, and reputational cost,
  which are real but which we have no sourced India-specific figure for, so we don't quote
  a number for them.
- **ILO: occupational accidents and diseases cost ~4% of global GDP annually** (~US$2.8
  trillion) in direct and indirect costs — the global benchmark for "this is not a rounding
  error," used to frame scale, not as an India-specific number.
- **228,585 registered factories in India** at end of 2022 (Labour Bureau, Ministry of
  Labour & Employment) — the base for addressable-market sizing below.

## Anchor incidents (use these, not hypotheticals)

- **Visakhapatnam Steel Plant, Jan 2025** — 8 dead, coke-oven explosion. Gas sensors had
  signal; no intelligence layer connected it to action. This is literally the problem
  statement's own example.
- **Sigachi Industries, Telangana, 30 Jun 2025** — 46 dead, 33 injured, dust explosion at
  a microcrystalline cellulose plant. Officials publicly questioned inspection/enforcement
  adequacy.

Both are exactly the "compound risk no single sensor would flag alone" failure mode this
system targets — a hazardous condition co-occurring with normal operating activity, with no
layer fusing the two before it was too late.

## Addressable market (our estimate, not sourced)

| Tier | Definition | Count (our estimate) | Basis |
|---|---|---|---|
| TAM | All registered factories in India | 228,585 | Labour Bureau, end-2022 |
| SAM | Factories engaged in a declared hazardous process (Factories Act Ch. IVA) — the segment with confined spaces, flammable/toxic materials, permit-to-work regimes | ~15,000-20,000 (our estimate, no national count found) | Order-of-magnitude only — flag this as an estimate if asked, we found no published count of hazardous-process-registered factories |
| SOM (3-year) | Large single-site plants (steel, oil & gas, chemicals, mining) able to afford and operationally justify a dedicated safety-intelligence platform | 300-500 facilities | Our estimate, consistent with the scale of CPSEs + large private plants (SAIL, IOCL, ONGC, Tata Steel, Vedanta-scale sites, etc.) |

## Illustrative unit economics (our assumption, for the pitch — not a costed quote)

- SaaS pricing: **₹15-25 lakh/year per large facility** (illustrative — comparable
  industrial-SaaS deployments in this space are typically priced per-site, not per-seat).
  At 300 facilities (SOM), that's a **₹45-75 crore/year** addressable revenue pool —
  state this as illustrative market sizing, not a forecast.
- Payback framing for a customer: **one prevented fatality (₹15-25+ lakh in direct
  compensation alone, before legal/downtime/reputational cost) covers more than a full
  year of licensing** at the assumed price point. This is the line to say on stage — it's
  defensible because both halves (compensation figure, price point) are stated as what
  they are: one sourced, one assumed.

## What NOT to claim

- Don't say "saves X lives/year" as a hard number — we have no causal estimate, only the
  mechanism (compound detection precision 0.91 vs baseline 0.37, see eval/results/metrics.json).
- Don't quote a total economic cost of accidents specific to India — we found no sourced
  India-specific GDP-loss figure, only the global ILO 4% figure. Don't let "4% of GDP" be
  misheard as India's number on stage; say "globally" explicitly.
- Don't claim the lead-time advantage — see eval/metrics.py's documented finding: once
  permit/presence visibility is properly time-gated, the GNN does not detect earlier than
  the naive baseline on this benchmark. The honest claims are precision and zone-localization.
