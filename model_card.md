# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RFC Random Forest Classifier was used mainly with its default settings 

## Intended Use
Predict salary of individuals according to some social criteria such as race, education and many others

## Training Data
The data used are collected from a census and can be found with this [link](https://archive.ics.uci.edu/ml/datasets/census+income). The ratio 80% of this data was used for the model train
We used two encoders LabelBinarizer for the label and OneHotEncoder for the categorical features

## Evaluation Data
The remaining 20% of the data was used for the evaluation

## Metrics
The metrics used are: Precision, Recall, f1-score and fbeta score. 
their values for our model are: Precision=0.6709, Recall=0.5737, f1-score=0.6155, fbeta=0.6185
As a reminder: Precision=TP/(TP+FP), Recall=TP/(TP+FN), f1-score=2*Precision*recall/(Precision+recall)
The fbeta score is the weighted harmonic mean of precision and recall.

## Ethical Considerations
This type of tool is interesting for highlighting the level of impact on the discriminated people,
but be careful so that it is not precisely used to discriminate in salaries.
We have to be careful of results that are biased for less represented groups like Amer-Indian-Eskimo and other for race feature

## Caveats and Recommendations
The imbalance of data encourages us to seek more data to remedy this problem. 
With the aim of improving resultsIt is possible to use dataset balancing methods like SMOTE: method of over-sampling.

