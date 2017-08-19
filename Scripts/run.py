
import os
s = "".join(("/Users/antoshkaplus/Documents/Programming",
     "/Contests/Kaggle/DigitRecognizer/DerivedData/DigitRecognizer/Build/Products/Release/DigitRecognizer"))
os.system(s + " -training_set train.csv -test_set test.csv -result res.txt")

