We have a seperate branch for each task:

- ```carexpert``` for the Race Car challenge
- ```tumor/nnunet``` for the TS challenge
- ```emergency-healthcare-rag``` for the Healthcare RAG challenge

User guides:

### Race Car
For evaluation we used the model models/LaneShift.py, which has the "return_action" function interface as the ambolt AI team set it up. The api.py file is set up to use the model correctly, and expose the prediction endpoint, so from there you should be able to see how the model is used if there are any doubts.

### TS challenge
See the README_TS.md file in the branch.
The model is too big to upload but the readme descibes how to train an identical model invluding the entire pipeline. We can send the model upon request if needed.

### Healthcare RAG
We mainly used UCloud for running our code. The API can be run in the ```emergency-healthcare-rag/``` folder with ```python api.py --no-use-condensed-topics --threshold NA``` with our settings used for the evaluation run.


Thank you for arranging this competition!
 - Benjamin, Elias, Jonathan, Lucas, Oscar, and Viktor
