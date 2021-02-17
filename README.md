# NLP project on Research papers similarity



## Dependencies

> Python 3.7-> Please use virtual env to isolate your python development environments https://packaging.python.org/guides/installing-using-pip-and-virtual-environments

## Getting started
* Clone repo
`git clone https://github.com/eendee/resarch-paper-sim.git`

* Start virtual environment
 *On Linux and MacOS*
  `source bin/activate`
 *On Windows*
  `Scripts\activate.bat`
* Install requirements.txt
  `pip install -r requirements.txt`


## Dataset
`/dataset` contains two files:
1. corpus_dict.pkl : a dictionary of 1000 papers, the dictionary key is the paper id and the value is a list of paragraphs of the body of the paper 

2. corpus_meta.pkl: A dataframe of the metadata of the papers in (1), the two can be matched with their ids for lookup purposes.

