# Newyork Taxi Trip 

> Please check [artifacts](./artifacts/) for the inputs and outputs.


## Objective
This project aims to tack three tasks:

`Question A`:
Calculate on the entire dataset the 5th, 50th and 95th percentiles (q05, q50, q95) on the dataset
values: 'fare_amount ' , ' tip_amount ' and ' total_amount ' ; divided according to the ' VendorID ',
' passenger_count ' and ' payment_type ' fields
The calculation output must be a dataFrame to be exported in CSV format organized with:
Columns: field name (on which the percentile is calculated) + “_p_” + percentile threshold
Rows (index): grouping field name + ” _” + value of the group on which the percentile calculation is performed
`Question A.1 (optional)`:
Calculate the percentiles as reported for question A also for the dataset divided by trip_distance if
>2.8 or <=2.8 and add the calculated values to the dataFrame with the logic reported in question A


`Question B`:
Generate an ML model for estimating the " total_amount " based on the variables (as input to the
model): ' VendorID ','passenger_count ' ,'payment_type ' ,' trip_distance '
It is possible to independently define the methodology and the selection and split process of the
reference dataset for training, testing and verification of the model (kf, random, train -test- valid )
(optional) For model optimization it is recommended to calculate the RMSE on the selected partial
test dataset
Export the generated model to file (ie via pickle, json …)
*The quality assessment of the generated model will be verified through the calculation of the RMSE on a test dataset equivalent to the one used by the user in terms of format and compatible in terms of number (but not provided)
**The user is given the right to use a different ML model from those present in the sklearn libraries, but the generated model must be exportable, and the user must indicate the name and version of the library used (for calculation of the RMSE on the new test dataset)



## How to execute?
First of all you need to create an environment using conda as below:

```
conda env create -f environment.yml
conda activate taxi
python -m ipykernel install --user --name "taxi"
```


There are two approaches to execute tasks:
- Using Notebooks: there are some notebooks in [notebook](./notebook/) folder which has two sub-folder to solve each question.
- Modular programming: to execute using this approach you need to execute commands below in your terminal.

```
pip install -e .
taxi
```