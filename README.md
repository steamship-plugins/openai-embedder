# OpenAI Embedder Plugin - Steamship

This project contains a Steamship Tagger plugin that enables embedding with OpenAI's models.

## Configuration

This plugin must be configured with the following fields:

* `model` - The model, listed in the [OpenAI Documentation](https://studio.oneai.com/docs?api=Pipeline+API&item=Expected+Input+Format&accordion=Introduction%2CPipeline+API%2CNode.js+SDK+Reference%2CClustering+API).
* `dimensionality` - Look up from the model family below.

OpenAI supports four families of embedding models for different functionalities: text search, text similarity and code search. 
Each family includes up to four models on a spectrum of capability:

* Ada (1024 dimensions)
* Babbage (2048 dimensions)
* Curie (4096 dimensions)
* Davinci (12288 dimensions)

Within those model families you can select:

* `text-similarity-ada-001`
* `text-similarity-babbage-001`
* `text-similarity-curie-001`
* `text-similarity-davinci-001`
* `text-search-ada-doc-001`
* `text-search-ada-query-001`
* `text-search-babbage-doc-001`
* `text-search-babbage-query-001`
* `text-search-curie-doc-001`
* `text-search-curie-query-001`
* `text-search-davinci-doc-001`
* `text-search-davinci-query-001`
* `code-search-ada-code-001`
* `code-search-ada-text-001`
* `code-search-babbage-code-001`
* `code-search-babbage-text-001`

