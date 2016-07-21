# FeatureRobustness:

Understanding the impact of different factors on the features and their uses in ML applied to photometric SN Classification

[![Join the chat at https://gitter.im/rbiswas4/FeatureRobustness](https://badges.gitter.im/rbiswas4/FeatureRobustness.svg)](https://gitter.im/rbiswas4/FeatureRobustness?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


## Background

In photometric SN classification, we are trying to start off with some 'training data' and summarize the properties of different known types of SN in the training data, either in terms of models (template fitting based methods), explicitly stated features or perhaps leaving the features implicitly defined by the labels in he training set.

We know that this is a complicated problem in general, so stepping back let us imagine a simpler version of the problem. Consider a single rest frame parametrized SNIa and a single rest frame parametrized SNIb/SNIc observed with high SNR with very good temporal sampling in all bands.  Using a single such Ia and Ib, it should be easy to distinguish all exact repilicas of these SN.

However, the real situation is difficult because of several different factors both in training and test sample (which may not match up):
- Diversity: All SNIa or SNIb do not look the same. Part of this can be parametrized using models, but there is also diversity of these SN beyond the model description.
- Different time sampling : The supernovae go off at different times and they have to be translated back to a standard time interval. Depending on the prescription of the transformation, this could also depends on the redshift, and the times of observation. 
- Different Sky Noise levels : Different levels of noise on Observations 
- Different redshifts : 

This gives us a stochastic distribution of features in any data sample. As a result, it is unclear how to define a 'distance' like measure between two points. As has been discussed, in talks both here and previously, t-SNE attempts to give us such a quantity. One could also try to use other similar measures such as a Kulback Leibler Divergence between the distributions of features in a stochastic sample.

The idea here is the following:
- Let us use a concrete measure of distance, such as t-SNE (since we have a code which gives this for features), but if others happen to have similar measures coded up. (@rbiswas4 can help in making sure we have some of these)
- Let us start from keeping a very simple set of models (SNIa and SNIbc) light curves at z = 0.3, with 'peaks' 'matched up' and Noise distribution defined by the LSST mean values. 
- Calculate features (based on snmachine) for these simplest cases (Hopefully @MichelleLochner can find some time to help with this)
- Increase complexity on the time sampling and redshift changes axis. 
- Play with subsets of the features
- Play with changing the training fraction

This should show that the t-SNE of features change with changes in complexity and changes in the feature set. 
With increase in complexity, but fixed feature set, I am hoping to see the cluster of features representing a single class of objects to grow, with the separation between blobs to roughly remain the same. For a fixed complexity I am hoping to see the t-SNE separation to decrease while perhaps the cluster sizes also grow a little (but lesser effect) smaller. Maybe this is wrong, and others have a better idea.

## What does this help us with ?
Mostly, I hope to get some familiarity with the code and build some intuition. However, this may have some interesting possibilities
- Sometimes one has to pre-process the data for applying it to various algorithms. For example, we saw that all light curves had to be put on a standard interval (for example by taking differences with the first observed point) in time. Sometimes, it requires interpolation to a uniform grid. Are there significantly 'better' or 'worse' pre-processing methods we can think of with 'better' defined if the process leads to smalles size expansion of the t-SNE clusters? 
- Which of the contributions to complexity cause the most trouble. Are there ways of trying to help this?
- What are the impacts of the training fraction and decreasing/increasing feature sets?
