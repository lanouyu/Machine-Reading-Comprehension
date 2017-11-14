# Machine Reading Comprehension

The topic of this project is Machine Reading Comprehension (MRC), which should be built to read documents and answer questions about their content. The task is to solve cloze-style questions, that means answering the question with a noun/noun phrase to conform the meaning of document. The provided dataset has 21 consecutive sentences in one instance. While the 21st sentence is served as the question by replacing a single word with an anonymous placeholder token. Furthermore, 10 candidates are listed at the end of last line, only one of which is the correct answer.

## requirement:
* Python 2.7
* PyTorch
* Numpy
* cuda(opt.)

## Run:
./run.sh
you can modify the hyper-parameters, including the data path in the script.
