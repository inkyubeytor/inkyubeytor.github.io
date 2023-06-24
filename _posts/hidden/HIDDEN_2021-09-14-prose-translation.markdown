---
layout: post
title:  "Idea: Role Analysis Via Unsupervised Cross-Lingual Alignment"
categories: [Computational Literature]
tags: idea
---

A common task in analysis of fiction is the assignment of characters
to predefined narrative roles. These roles are chosen at the level of 
specificity needed for a downstream task, and range from generic roles such as 
"major character" to specific roles such as "informant."
This task is often performed through
the [analysis of character networks][1], which can be constructed either 
manually or via computational methods. However, this method of role 
analysis may fail to leverage a large amount of information present in 
the text due to the intermediate construction of the character network.

I propose applying unsupervised cross-lingual alignment techniques to 
distributional word embeddings trained on different fictional texts, using the 
discovered alignments between vectors representing character names to generate 
and assign narrative roles across multiple texts.

# Background: Word Embeddings and Multilingual Alignment

Word embeddings (vectors representing words) are often constructed upon the
principle of [distributional semantics][4], that the meaning of a word is 
defined by the contexts in which it occurs. With models such as Word2Vec,
larger contexts tend towards more topical or semantic meanings, while smaller
contexts tend towards more syntactic meanings.

[Recent work][2] in multilingual NLP deals with aligning sets of monolingual word 
vectors to identify cross-lingual word associations. [Other work][3] in 
monolingual NLP leverages machine translation techniques to work with different
corpora from the same language with different distributional properties, 
treating sets of differently polarized political commentary as coming from 
different languages.

# Application to Role Analysis

Fiction works, especially speculative fiction, can differ wildly in the words
they use (e.g. spell names in _Harry Potter_), what they intend those words to 
mean (e.g. "cape" in _Worm_), and how often they use those words (e.g. "ring" 
in _The Lord of the Rings_). By acknowledging these distributional differences
and treating different works of fiction as coming from different languages, we
can then apply word embedding alignment methods to identify associations 
between differing words. 

If working with monolingual data, we can expect high alignment between words 
occurring in similar contexts across multiple works. As this is likely much of
the vocabulary, our overall alignment will likely be of high quality. We can 
then look at strong associations between character name embeddings to create
groupings of characters that share similar roles across works, assigning these
groups semantic labels if applicable.

Possible hyperparameters to tune include the threshold on required similarity 
for membership in the group and the window size used for the construction of 
word embeddings. Intuitively, higher thresholds would increase the specificity
of role groups, and lower thresholds would increase generality. 
Larger windows may increase the generality of role groups, as
distributions may become more similar with larger contexts. However, small 
windows may inadvertently increase generality by capturing the syntactic usage
of a name rather than the distributional properties of the character's 
occurrences.

Should this project be successful, future work can extend this methodology 
beyond character roles, looking at associations between other 
narrative-specific terminology or even at words or phrases of constructed 
languages embedded in the work.

[1]: https://arxiv.org/pdf/1907.02704.pdf
[2]: https://arxiv.org/pdf/1809.03633.pdf
[3]: https://arxiv.org/pdf/2010.02339.pdf
[4]: https://aclanthology.org/W17-0239.pdf