# CHANGELOG

<!-- version list -->

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
