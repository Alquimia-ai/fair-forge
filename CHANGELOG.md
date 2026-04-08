# CHANGELOG

<!-- version list -->

## v3.0.0-b.5 (2026-04-08)

### Bug Fixes

- **ci**: Update mintlify CLI to version with validate command
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **core**: Use statistical_mode rng and guard zero-weight in bootstrap
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **core**: Validate assigned weights sum in _resolve_weights
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **docs**: Address PR review comments ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **guardians**: Handle API error responses in OpenAIGuardianProvider
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Fallback to structured_response for chat history assistant entry
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Make judge evaluations atomic and add opt-in chat history
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Move ChatPromptTemplate to module level and update structured mode tests
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Truncate retry log to avoid leaking sensitive data
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Skip single-assistant blocks and document king-of-the-hill semantics
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Warn on mixed-language datasets and document limitation
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **schemas**: Enforce non-negative constraint on Batch.weight
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Chores

- Add LaTeX build artifacts to .gitignore ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- Remove cloud deps group and fix notebook cell source format
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- Remove dep aenum ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **ci**: Fix sync-develop job output and auto-resolve conflicts with main
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **deps**: Update uv.lock ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **storage**: Remove storage module and docs
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Continuous Integration

- Downgrade node to 18 to fix katex __VERSION__ error in mintlify
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- Pin mintlify to 4.0.5 to avoid katex __VERSION__ build bug
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- Restrict release trigger to fair_forge directory changes
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Documentation

- **core-concepts**: Clarify BestOf granularity behavior with stream_batches
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **core-concepts**: Rewrite retriever docs and add streaming retrievers page
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **examples**: Note json.load memory limitation in streaming retrievers
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **examples**: Use portable PyPI install in context notebook
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add vision metrics documentation and test fixtures
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Update agentic docs to reflect global pass@K and new API
  ([`910808e`](https://github.com/Alquimia-ai/fair-forge/commit/910808ecb09cfb12efa6fa72810a06df23d8b595))

- **paper**: Add agentic metric white paper with experiments and results
  ([`3bf24e2`](https://github.com/Alquimia-ai/fair-forge/commit/3bf24e2d42093fdf57879dab9119aa112ef1e2c8))

- **paper**: Address review feedback on Prompt Evaluator paper
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **paper**: Address reviewer feedback on notation and consistency
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **paper**: Clarify RM and PP as theoretical components not yet implemented
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **papers**: Add context adherence metrics paper
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **papers**: Add English LaTeX paper for PromptEvaluator metric design
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **papers**: Add LaTeX preamble, bibliography and ignore build artifacts
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **papers**: Add preamble and proposed experiments section to regulatory paper
  ([`aeb66db`](https://github.com/Alquimia-ai/fair-forge/commit/aeb66db8b23889985425d7de78596bb43a1b9f3f))

- **papers**: Add references section (.bib) to regulatory paper
  ([`36f4204`](https://github.com/Alquimia-ai/fair-forge/commit/36f42045709f42df1386fe8d0f718d9bf2f0030f))

- **papers**: Add regulatory compliance metric paper
  ([`5e9e382`](https://github.com/Alquimia-ai/fair-forge/commit/5e9e382f5e7d745c97f705608a587a2fdb3248ff))

- **plan**: Add PromptEvaluator dataset implementation plan
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **spec**: Add PromptEvaluator dataset design spec
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Features

- **core**: Add streaming dataset support with iteration level strategy
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **examples**: Add Heretic generator notebook for vLLM HuggingFace endpoint
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **examples**: Add streaming retrievers and context notebook streaming section
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add PromptEvaluator metric ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add statistical mode support to bias and agentic
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add statistical mode to conversational, context, regulatory
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add VisionSimilarity and VisionHallucination metrics for VLM evaluation
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Compute pass@K and pass^K globally across all conversations
  ([`dbb265a`](https://github.com/Alquimia-ai/fair-forge/commit/dbb265a3cf6506cf31820131745063e2a7b92f5f))

- **metrics**: Redesign PromptEvaluator as compound distributional metric
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Replace vision hallucination metrics with VisionSimilarity and VisionHallucination
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **paper**: Rewrite experiments section and add behavioral case study
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **prompt-optimizer**: Add GEPA and MIPROv2 prompt optimization module
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **prompt-optimizer**: Expose tips and proposal prompts as optional args in MIPROv2
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **schemas**: Add IterationLevel, SessionMetadata, and StreamedBatch types
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Performance Improvements

- **core**: Vectorize bayesian bootstrap sampling in _aggregate_scores
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Refactoring

- Move experiments directory outside fair_forge package
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **core**: Extract Embedder and Reranker ABCs for dependency inversion
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **core**: Simplify iteration_level handling and remove retriever_cls
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Add chat_history property for backward compatibility
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **llm**: Migrate Judge to create_agent with ProviderStrategy
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Apply Strategy and Adapter patterns to vision metrics
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

### Testing

- **core**: Add streaming mode and iteration level detection tests
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **metrics**: Add missing strict param to Judge initialization assertions
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))

- **toxicity**: Call on_process_complete after batch in unit tests
  ([#56](https://github.com/Alquimia-ai/fair-forge/pull/56),
  [`3fcd87e`](https://github.com/Alquimia-ai/fair-forge/commit/3fcd87e5725f4ed392001aea5a9e871af676e4fe))


## v3.0.0-b.4 (2026-04-01)

### Bug Fixes

- **llm**: Fallback to structured_response for chat history assistant entry
  ([`66fd839`](https://github.com/Alquimia-ai/fair-forge/commit/66fd8394b89bbca65d2e79819f9a16c507e0c597))

- **llm**: Make judge evaluations atomic and add opt-in chat history
  ([`a19fb46`](https://github.com/Alquimia-ai/fair-forge/commit/a19fb4676eecf5a3ab2a72c53313309fcd5a3282))

- **llm**: Truncate retry log to avoid leaking sensitive data
  ([`3879d2e`](https://github.com/Alquimia-ai/fair-forge/commit/3879d2e53fb848d1a342d5d97e704d55794c4dec))

### Documentation

- **examples**: Use portable PyPI install in context notebook
  ([`6457f2a`](https://github.com/Alquimia-ai/fair-forge/commit/6457f2a760a629471fba189ba71f7a44f044a5fd))

- **papers**: Add LaTeX preamble, bibliography and ignore build artifacts
  ([`7c381f2`](https://github.com/Alquimia-ai/fair-forge/commit/7c381f2ab4d0a9173d3abfcea9b80dcf6240313e))

### Refactoring

- **llm**: Add chat_history property for backward compatibility
  ([`dd5abef`](https://github.com/Alquimia-ai/fair-forge/commit/dd5abefadfb42b3b4a522f14f72d8f612935cf92))


## v3.0.0-b.3 (2026-04-01)

### Chores

- Add LaTeX build artifacts to .gitignore ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

### Documentation

- **paper**: Address review feedback on Prompt Evaluator paper
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **paper**: Address reviewer feedback on notation and consistency
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **paper**: Clarify RM and PP as theoretical components not yet implemented
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **papers**: Add context adherence metrics paper
  ([`33a4c57`](https://github.com/Alquimia-ai/fair-forge/commit/33a4c57531dbe18fedc8d77a8ec98e9e8aec8c1e))

- **papers**: Add English LaTeX paper for PromptEvaluator metric design
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **plan**: Add PromptEvaluator dataset implementation plan
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **spec**: Add PromptEvaluator dataset design spec
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

### Features

- **metrics**: Add PromptEvaluator metric ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **metrics**: Redesign PromptEvaluator as compound distributional metric
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))

- **paper**: Rewrite experiments section and add behavioral case study
  ([#51](https://github.com/Alquimia-ai/fair-forge/pull/51),
  [`38265be`](https://github.com/Alquimia-ai/fair-forge/commit/38265beb1dfa77c288e73de5a949464e12b222a2))


## v3.0.0-b.2 (2026-03-18)

### Documentation

- **metrics**: Add vision metrics documentation and test fixtures
  ([`16313bc`](https://github.com/Alquimia-ai/fair-forge/commit/16313bc021b6e88a2f1a8bef665415fbd1c1f792))

### Features

- **metrics**: Add VisionSimilarity and VisionHallucination metrics for VLM evaluation
  ([`ca50030`](https://github.com/Alquimia-ai/fair-forge/commit/ca500307702fc5471b7c2f6429c72aee379432e3))

- **metrics**: Replace vision hallucination metrics with VisionSimilarity and VisionHallucination
  ([`10e9f88`](https://github.com/Alquimia-ai/fair-forge/commit/10e9f88951e18291bc4242236cffac5cf6eaef2e))

### Refactoring

- **metrics**: Apply Strategy and Adapter patterns to vision metrics
  ([`e0a7c8d`](https://github.com/Alquimia-ai/fair-forge/commit/e0a7c8d423dc7c0ad02d011367cee6de1f2c34ea))


## v3.0.0-b.1 (2026-03-12)

### Chores

- **ci**: Fix sync-develop job output and auto-resolve conflicts with main
  ([`3ab30ed`](https://github.com/Alquimia-ai/fair-forge/commit/3ab30ed326b0d6192f30a400b92a04643bfd1e80))

### Features

- **prompt-optimizer**: Add GEPA and MIPROv2 prompt optimization module
  ([`9dd083b`](https://github.com/Alquimia-ai/fair-forge/commit/9dd083beceb65f08b014e8a45fdab4ab829c3217))

- **prompt-optimizer**: Expose tips and proposal prompts as optional args in MIPROv2
  ([`e662879`](https://github.com/Alquimia-ai/fair-forge/commit/e6628799e9107dde534b0d8592972e67ad8deae5))


## v2.0.0 (2026-03-12)

### Bug Fixes

- **ci**: Update mintlify CLI to version with validate command
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core**: Use statistical_mode rng and guard zero-weight in bootstrap
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core**: Validate assigned weights sum in _resolve_weights
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **docs**: Address PR review comments ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **guardians**: Handle API error responses in OpenAIGuardianProvider
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **llm**: Move ChatPromptTemplate to module level and update structured mode tests
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **metrics**: Skip single-assistant blocks and document king-of-the-hill semantics
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **metrics**: Warn on mixed-language datasets and document limitation
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **schemas**: Enforce non-negative constraint on Batch.weight
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Chores

- Remove cloud deps group and fix notebook cell source format
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- Remove dep aenum ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **deps**: Update uv.lock ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **storage**: Remove storage module and docs
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Continuous Integration

- Downgrade node to 18 to fix katex __VERSION__ error in mintlify
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- Pin mintlify to 4.0.5 to avoid katex __VERSION__ build bug
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- Restrict release trigger to fair_forge directory changes
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Documentation

- **core-concepts**: Clarify BestOf granularity behavior with stream_batches
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core-concepts**: Rewrite retriever docs and add streaming retrievers page
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **examples**: Note json.load memory limitation in streaming retrievers
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Features

- Statistical mode, streaming datasets & LLM refactor
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core**: Add streaming dataset support with iteration level strategy
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **examples**: Add Heretic generator notebook for vLLM HuggingFace endpoint
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **examples**: Add streaming retrievers and context notebook streaming section
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **metrics**: Add statistical mode support to bias and agentic
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **metrics**: Add statistical mode to conversational, context, regulatory
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **schemas**: Add IterationLevel, SessionMetadata, and StreamedBatch types
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Performance Improvements

- **core**: Vectorize bayesian bootstrap sampling in _aggregate_scores
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Refactoring

- Move experiments directory outside fair_forge package
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core**: Extract Embedder and Reranker ABCs for dependency inversion
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **core**: Simplify iteration_level handling and remove retriever_cls
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **llm**: Migrate Judge to create_agent with ProviderStrategy
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

### Testing

- **core**: Add streaming mode and iteration level detection tests
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **metrics**: Add missing strict param to Judge initialization assertions
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))

- **toxicity**: Call on_process_complete after batch in unit tests
  ([#47](https://github.com/Alquimia-ai/fair-forge/pull/47),
  [`9d0431b`](https://github.com/Alquimia-ai/fair-forge/commit/9d0431b8d37f5f6dafa5408be8602341db7531ba))


## v2.0.0-b.3 (2026-03-10)

### Bug Fixes

- **ci**: Update mintlify CLI to version with validate command
  ([`5496680`](https://github.com/Alquimia-ai/fair-forge/commit/5496680884d9e68fa15ba884f2912d9d00e508fb))

- **docs**: Address PR review comments
  ([`241f277`](https://github.com/Alquimia-ai/fair-forge/commit/241f277851a0e0f9b9f7c80e3d825ed4f7390f73))

- **guardians**: Handle API error responses in OpenAIGuardianProvider
  ([`795e028`](https://github.com/Alquimia-ai/fair-forge/commit/795e028a1a220bbaa33105f01b00e61ab7c0cabe))

- **llm**: Move ChatPromptTemplate to module level and update structured mode tests
  ([`875d278`](https://github.com/Alquimia-ai/fair-forge/commit/875d27889c572f05582f6d67a856d203587c2097))

### Continuous Integration

- Restrict release trigger to fair_forge directory changes
  ([`f0d592e`](https://github.com/Alquimia-ai/fair-forge/commit/f0d592eeb041b027b8c53b348cb09a8598ea7749))

### Refactoring

- Move experiments directory outside fair_forge package
  ([`27a2874`](https://github.com/Alquimia-ai/fair-forge/commit/27a2874fa2f0a61be7c246415fe864a6a6e3fda9))

- **core**: Extract Embedder and Reranker ABCs for dependency inversion
  ([`ae09be1`](https://github.com/Alquimia-ai/fair-forge/commit/ae09be166382670f8c2af66f6482be62aeb792e3))

- **llm**: Migrate Judge to create_agent with ProviderStrategy
  ([`2411cbd`](https://github.com/Alquimia-ai/fair-forge/commit/2411cbdca43fb30ee63a02155624ae7639be9b6c))

### Testing

- **metrics**: Add missing strict param to Judge initialization assertions
  ([`0d18989`](https://github.com/Alquimia-ai/fair-forge/commit/0d18989611a8dd4bc504eccc7722a652505705c2))


## v2.0.0-b.2 (2026-03-06)

### Features

- **examples**: Add Heretic generator notebook for vLLM HuggingFace endpoint
  ([`0fcaeb1`](https://github.com/Alquimia-ai/fair-forge/commit/0fcaeb12f58e6d8d6fb2d62bcd18469bc24f1c08))


## v2.0.0-b.1 (2026-03-05)

### Bug Fixes

- **core**: Use statistical_mode rng and guard zero-weight in bootstrap
  ([`4ae7810`](https://github.com/Alquimia-ai/fair-forge/commit/4ae78104c4c9c25164e69502a5929d651dc79de6))

- **core**: Validate assigned weights sum in _resolve_weights
  ([`3e80edf`](https://github.com/Alquimia-ai/fair-forge/commit/3e80edf5e0f2fe5c4a7bc51573a423daefdbbcd5))

- **schemas**: Enforce non-negative constraint on Batch.weight
  ([`9a48fb9`](https://github.com/Alquimia-ai/fair-forge/commit/9a48fb91a12b3a8f2114ad040ebff44d1e5a9aa1))

### Chores

- **deps**: Update uv.lock
  ([`c2df7b3`](https://github.com/Alquimia-ai/fair-forge/commit/c2df7b33f57d0bd702729460c0e9ec55b9efc676))

### Continuous Integration

- Downgrade node to 18 to fix katex __VERSION__ error in mintlify
  ([`b5e1c0e`](https://github.com/Alquimia-ai/fair-forge/commit/b5e1c0e6afa622766b1f2d265b17098e96004680))

- Pin mintlify to 4.0.5 to avoid katex __VERSION__ build bug
  ([`8421784`](https://github.com/Alquimia-ai/fair-forge/commit/8421784452aa329cf0f189d7ae85f7bd46e83f4a))

### Features

- **metrics**: Add statistical mode support to bias and agentic
  ([`5d7e760`](https://github.com/Alquimia-ai/fair-forge/commit/5d7e76015a7d940a3acfeaffb993fc92a1724aaf))

- **metrics**: Add statistical mode to conversational, context, regulatory
  ([`7102533`](https://github.com/Alquimia-ai/fair-forge/commit/7102533e4568e1d3001c1bef377a28f7eaa69f34))

### Performance Improvements

- **core**: Vectorize bayesian bootstrap sampling in _aggregate_scores
  ([`0083643`](https://github.com/Alquimia-ai/fair-forge/commit/008364309a30862fb28b92c6dd9a338e96d1595e))


## v1.3.0-b.1 (2026-02-27)

### Bug Fixes

- **metrics**: Skip single-assistant blocks and document king-of-the-hill semantics
  ([`c57a827`](https://github.com/Alquimia-ai/fair-forge/commit/c57a827c6a8e729459ffb862fb9851cc07f5dba7))

- **metrics**: Warn on mixed-language datasets and document limitation
  ([`52c7d95`](https://github.com/Alquimia-ai/fair-forge/commit/52c7d95c7b7aaafe036a0ad5a33e03a759d0b15e))

### Chores

- Remove cloud deps group and fix notebook cell source format
  ([`e5b6f06`](https://github.com/Alquimia-ai/fair-forge/commit/e5b6f06c571a6a6e3aeafb9c142878aa519fa3d0))

- Remove dep aenum
  ([`6d07e37`](https://github.com/Alquimia-ai/fair-forge/commit/6d07e37fa921f8ebdbe8e72d6e93a5c12acd6df3))

- **storage**: Remove storage module and docs
  ([`174d31e`](https://github.com/Alquimia-ai/fair-forge/commit/174d31e833f8577c76a2ff59eeabae28c670bce1))

### Continuous Integration

- **release**: Skip release on docs-only changes
  ([`df28fd1`](https://github.com/Alquimia-ai/fair-forge/commit/df28fd13531696f7d0944cff870c33166041250e))

### Documentation

- Add fair forge logo and update branding
  ([`5f014c2`](https://github.com/Alquimia-ai/fair-forge/commit/5f014c2505144e39ef082785dd5ee5ecd84135e5))

- **core-concepts**: Clarify BestOf granularity behavior with stream_batches
  ([`5583f2c`](https://github.com/Alquimia-ai/fair-forge/commit/5583f2ce7aba80c2ae0bdc0c0e521323d3751623))

- **core-concepts**: Rewrite retriever docs and add streaming retrievers page
  ([`8a7731a`](https://github.com/Alquimia-ai/fair-forge/commit/8a7731a369f366ff69126ac13cf59a7b9f49c83c))

- **examples**: Note json.load memory limitation in streaming retrievers
  ([`adb44a1`](https://github.com/Alquimia-ai/fair-forge/commit/adb44a1fc31c3787e3e2cc889562abd2415106aa))

### Features

- **core**: Add streaming dataset support with iteration level strategy
  ([`d76e461`](https://github.com/Alquimia-ai/fair-forge/commit/d76e461e43e58efe11de38b4350e9b1d8b1fff26))

- **examples**: Add streaming retrievers and context notebook streaming section
  ([`b092b2d`](https://github.com/Alquimia-ai/fair-forge/commit/b092b2d80cf2aef6c9afc952af166f9f2baeff45))

- **schemas**: Add IterationLevel, SessionMetadata, and StreamedBatch types
  ([`8eb8a02`](https://github.com/Alquimia-ai/fair-forge/commit/8eb8a02aaa0640f18a4232d6b4ef153a78cf9733))

### Refactoring

- **core**: Simplify iteration_level handling and remove retriever_cls
  ([`3b71c13`](https://github.com/Alquimia-ai/fair-forge/commit/3b71c133a192361409c4d6a1f2ec8085d1714b7a))

### Testing

- **core**: Add streaming mode and iteration level detection tests
  ([`e6a1f96`](https://github.com/Alquimia-ai/fair-forge/commit/e6a1f96850773077ded4296ce5a5344da05990cb))

- **toxicity**: Call on_process_complete after batch in unit tests
  ([`58e8476`](https://github.com/Alquimia-ai/fair-forge/commit/58e84768cf40871d2b627d242a9f4d7edff71579))


## v1.2.0 (2026-02-23)

### Continuous Integration

- **release**: Skip release on docs-only changes
  ([`df28fd1`](https://github.com/Alquimia-ai/fair-forge/commit/df28fd13531696f7d0944cff870c33166041250e))

### Documentation

- Add fair forge logo and update branding
  ([`5f014c2`](https://github.com/Alquimia-ai/fair-forge/commit/5f014c2505144e39ef082785dd5ee5ecd84135e5))



## v1.2.0-b.7 (2026-02-23)

### Bug Fixes

- **docs**: Use mintlify dark class selectors instead of prefers-color-scheme
  ([`c349eae`](https://github.com/Alquimia-ai/fair-forge/commit/c349eae3fc9585ca138909980ebdacd49992c76a))


## v1.2.0-b.6 (2026-02-23)

### Bug Fixes

- **regulatory**: Address PR review comments
  ([`42f54d6`](https://github.com/Alquimia-ai/fair-forge/commit/42f54d66171cb7ff4e29ae10d64bd687d29be415))

- **regulatory**: Set compliance_threshold in unit tests
  ([`c4f7c48`](https://github.com/Alquimia-ai/fair-forge/commit/c4f7c48394921008f457c48df20afbf00b8f0ced))

### Documentation

- **regulatory**: Add documentation for regulatory compliance metric
  ([`3814613`](https://github.com/Alquimia-ai/fair-forge/commit/3814613a4bd8b4728a1af79d423f8ec7e5cb1fd0))

### Features

- **regulatory**: Add regulatory compliance metric
  ([`3472289`](https://github.com/Alquimia-ai/fair-forge/commit/3472289c2b6a004dbe7a8ff28961a5ca58b6e737))

- **skills**: Add metric-creator skill for Claude Code
  ([`0a87986`](https://github.com/Alquimia-ai/fair-forge/commit/0a87986b4018a6363a10d76703b6e24294b84f4f))


## v1.2.0-b.5 (2026-02-18)

### Bug Fixes

- **agentic**: Evaluate conversations as complete units with probabilistic formulas
  ([`95533bb`](https://github.com/Alquimia-ai/fair-forge/commit/95533bb78fc8dfff1ee1f4648f1b909c3056d313))

- **agentic**: Replace combinatorial pass@k with Bernoulli model and remove aggregate_metrics
  ([`2c68665`](https://github.com/Alquimia-ai/fair-forge/commit/2c686658e318c7c7e11ecca5cca3a832960ebb53))

- **agentic**: Update use_structured_output default to True in tests, docs, and lambda
  ([`ee1218e`](https://github.com/Alquimia-ai/fair-forge/commit/ee1218ec85eb3c0ca2ff19651d53a66de6c0d5b1))

- **docs**: Escape < and > characters in agentic MDX to fix parsing errors
  ([`cb2e864`](https://github.com/Alquimia-ai/fair-forge/commit/cb2e864e3b7cee68ea197d64c11d57a447e69541))

### Features

- **agentic**: Compute pass@k and pass^k per conversation with required k parameter
  ([`5dd936a`](https://github.com/Alquimia-ai/fair-forge/commit/5dd936a38dfb168166413625657a0c22d7a9f01b))


## v1.2.0-b.4 (2026-02-13)

### Bug Fixes

- **docs**: Use mintlify validate instead of npx validation package
  ([`35c8a39`](https://github.com/Alquimia-ai/fair-forge/commit/35c8a3981e1437b2d2241c5257b26df7ea689186))

### Build System

- Exclude experiments and examples from ruff
  ([`0f24b7b`](https://github.com/Alquimia-ai/fair-forge/commit/0f24b7b14c01d4fb867bbcdc843dc4a493aee83a))

### Chores

- Add .ruff_cache to gitignore
  ([`427ca60`](https://github.com/Alquimia-ai/fair-forge/commit/427ca6045b38bb7f4234c04decb41128bb4f4d68))

- New fair-forge version
  ([`3d9133a`](https://github.com/Alquimia-ai/fair-forge/commit/3d9133a7d38e76b3d36b252f8559ffa8a871e1f9))

- Remove legacy setup.py and MANIFEST.in
  ([`76aa482`](https://github.com/Alquimia-ai/fair-forge/commit/76aa4823b7372a7b4a7091a8f38c78acc7a7aa43))

### Continuous Integration

- **docs**: Add mintlify validation workflow
  ([`5552ab5`](https://github.com/Alquimia-ai/fair-forge/commit/5552ab5343326a76681cd10d90815402b18ed1b8))

### Documentation

- Add contributing section and remove explainability from readme
  ([`9f82b65`](https://github.com/Alquimia-ai/fair-forge/commit/9f82b6561d73d6a5c05c38fe8ed51c6844fa8869))

- Add SOLID, design patterns, and code smells guidelines to CLAUDE.md
  ([`06d1287`](https://github.com/Alquimia-ai/fair-forge/commit/06d12872134e7c2d62ec4142164b4f4ed157ed51))

- Replace uv pip install with uv add across all pages
  ([`cf62292`](https://github.com/Alquimia-ai/fair-forge/commit/cf62292c2e892f5e8ece3fb29077ff8f817fd9cd))

- Switch to aspen theme and add navigation group icons
  ([`741c3dc`](https://github.com/Alquimia-ai/fair-forge/commit/741c3dc4d352e48be81ca75b6c4b6c3db64c7a8f))


## v1.2.0-b.3 (2026-02-13)

### Features

- **explainability**: Add module with token attribution analysis
  ([`e5ac0ab`](https://github.com/Alquimia-ai/fair-forge/commit/e5ac0abb640659f2e7a087cea8bca36bf6019148))


## v1.2.0-b.2 (2026-02-12)

### Bug Fixes

- **agentic**: Evaluate tool correctness for all K responses
  ([`8fc4868`](https://github.com/Alquimia-ai/fair-forge/commit/8fc4868340bf096b08e9707290046df93f9eece3))

### Documentation

- **agentic**: Update documentation for tool_correctness_scores list
  ([`e7993b7`](https://github.com/Alquimia-ai/fair-forge/commit/e7993b72698da0c030809262298c4fe4e76f0bcf))


## v1.2.0-b.1 (2026-02-12)

### Build System

- **deps**: Upgrade langchain-core, nbconvert, and virtualenv
  ([`17e5c3a`](https://github.com/Alquimia-ai/fair-forge/commit/17e5c3a6a322c8936f9f53e6e1c5fabc61beb29a))

### Chores

- **ci**: Pass pypi token as cli argument
  ([`699813e`](https://github.com/Alquimia-ai/fair-forge/commit/699813e4bec7c2fc24341df3ce53c45e009206e9))

### Continuous Integration

- Unify release workflow with pre-release branch support
  ([`2d8e523`](https://github.com/Alquimia-ai/fair-forge/commit/2d8e52370722260c4c9a9db502790a65b2307951))

### Documentation

- Update logo url in readme
  ([`d04d699`](https://github.com/Alquimia-ai/fair-forge/commit/d04d699e3e2b176d3652fda79cc154978add0028))

- **agentic**: Add Mintlify documentation and examples
  ([`b084076`](https://github.com/Alquimia-ai/fair-forge/commit/b0840762ed803d33380e803059747abeaac579e0))

- **examples**: Add agentic and ground_truth_agentic fields to datasets
  ([`567da0f`](https://github.com/Alquimia-ai/fair-forge/commit/567da0f4a7b0f89bee8580a64e41ed1d531cd849))

### Features

- **experiments**: Add agentic metrics evaluation notebook
  ([`e374082`](https://github.com/Alquimia-ai/fair-forge/commit/e374082abbfb3d238e58a6a4c0217791d32ee3c7))

- **metrics**: Integrate agentic metric with LangChain support
  ([`aad6b9c`](https://github.com/Alquimia-ai/fair-forge/commit/aad6b9cae693a53ccf906177312cbef7bbc74d48))

### Testing

- **agentic**: Add comprehensive test suite for agentic metric
  ([`6fdd705`](https://github.com/Alquimia-ai/fair-forge/commit/6fdd7052aea753ec6ee1515569241e68fe90ed64))


## v1.1.0 (2026-02-02)

### Documentation

- Add BestOf Lambda deployment documentation
  ([`0da3de8`](https://github.com/Alquimia-ai/fair-forge/commit/0da3de8ec57d272498880acbd271bf6b8add66f5))

### Features

- **metrics**: Add AWS Lambda deployment for BestOf metric
  ([`bf87be3`](https://github.com/Alquimia-ai/fair-forge/commit/bf87be3ee729052c07eb9582699cc156c0707682))


## v1.0.1 (2026-01-30)

### Bug Fixes

- Judge tests verbosity
  ([`1c34f42`](https://github.com/Alquimia-ai/fair-forge/commit/1c34f42cf18a65625ebdcf99af833c95df426069))

- Update workflows gh
  ([`c314d84`](https://github.com/Alquimia-ai/fair-forge/commit/c314d845f17e9b2d01d41e9ceeac61a970fa0924))

- **llm**: Improve structured output with reasoning extraction and retry logic
  ([`7f8a624`](https://github.com/Alquimia-ai/fair-forge/commit/7f8a62402d612bfe609c81b85036a0c4412a038e))

### Build System

- **deps**: Upgrade langchain to 1.2+ for structured output support
  ([`cce3250`](https://github.com/Alquimia-ai/fair-forge/commit/cce3250f4181b426c9acf16e6dd0eed2a72416db))

### Chores

- Add develop as release branch
  ([`cf3b6a7`](https://github.com/Alquimia-ai/fair-forge/commit/cf3b6a7acd60b1d2d6fe8a86e75b06af368fb24f))


## v1.0.0 (2026-01-23)

- Initial Release
