# textprint

Different people write in different styles, choosing different words, syntactic structures, tones, etc. The goal of textprint is to use machine learning to "fingerprint" an author's writing style, making it possible to identify the author across different works.

# Data

The first real step will be getting enough data to train a model. The internet is rich with writing, so this might not seem hard. However, it is important to ensure that the writing can be *attributed* to a given person, making it possible to compare different works of the same author. Ideally, fetching would be done by downloading one author at a time, one text source at a time.

TODO:

 * Create a general fetching tool for source-agnostic text downloads
 * Fetch from online magazines/news sources
   * [The Guardian](https://www.theguardian.com/us)
   * [The New York Times](http://www.nytimes.com/)
 * Fetch from online forums
   * [Quora](https://www.quora.com)

# Model

**Current plan:** a character-based language model (an RNN) will take in a body of text and produce a single feature vector. The objective function will aim to bring feature vectors from the same author close together, while bringing feature vectors from different authors farther apart (in Euclidean space).

TODO:

 * Implement custom [sgd.SampleSet](https://godoc.org/github.com/unixpickle/sgd#SampleSet) for same-author and different-author samples.
 * Implement custom [sgd.Gradienter](https://godoc.org/github.com/unixpickle/sgd#Gradienter) for dealing with the custom samples.
