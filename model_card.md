# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RFC Random Forest Classifier was used mainly with its default settings 

## Intended Use
predict salary of individuals according to some social criteria such as race, education and many others

## Training Data
The data used are collected from a census and can be found with this [link](https://archive.ics.uci.edu/ml/datasets/census+income). The ratio 80% of this data was used for the model train

## Evaluation Data
The remaining 20% of the data was used for the evaluation

## Metrics
The metrics used are: Precision, Recall, f1-score and fbeta score

## Ethical Considerations
This type of tool is interesting for highlighting the level of impact on the discriminated people,
but be careful so that it is not precisely used to discriminate in salaries 

## Caveats and Recommendations
the imbalance of data encourages us to seek more data to remedy this problem

