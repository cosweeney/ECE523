# ECE 523 Term Project: Classifier Comparisons

## Connor Sweeney

--- 
### Required Python Libraries:
- numpy
- matplotlib
- tensorflow, keras
- scipy
- sklearn
- [tqdm](https://github.com/tqdm/tqdm) (used for progress bars)

Jupyter notebooks used for designing/testing the classifiers are listed as **CIFAR10_X.ipynb**. These were also used for making the plots seen in the term project report, which is listed as **"ECE523TermProjectReport.pdf"**. Some additional plots not included in the final paper are also present. 

The main file that can be used to run the classifiers as was done for the final classification test is **"Comparisons.py"** using the command `$ python Comparisons.py`. This file will state the accuracy for each classifier, and save a PNG of the confusion matrix plot made using sklearn's `ConfusionMatrixDisplay` method to the **/plots** folder for inspection. python 3.9.17 was used for all design and testing. 

If there are any issues, don't hesitate to let me know via cosweeney@arizona.edu. 
