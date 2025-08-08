We have a seperate branch for each task:

- ```carexpert``` for the Race Car challenge
- ```NA``` for the TS challenge
- ```emergency-healthcare-rag``` for the Healthcare RAG challenge

User guides:

### Race Car
For evaluation we used the model models/LaneShift.py, which has the "return_action" function interface as the ambolt AI team set it up. The api.py file is set up to use the model correctly, and expose the prediction endpoint, so from there you should be able to see how the model is used if there are any doubts.

### TS challenge


### Healthcare RAG
We mainly used UCloud for running our code. The API can be run in the ```emergency-healthcare-rag/``` folder with ```python api.py --no-use-condensed-topics --threshold NA``` with our settings used for the evaluation run.


Thank you for arranging this competition!
 - Benjamin, Elias, Jonathan, Lucas, Oscar, and Viktor
