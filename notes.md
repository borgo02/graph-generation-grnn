-deep graph encoders 
graph convolutions --> activation function --> reguralization/dropout -- >node outputs

generate graph similar to a given set of graphs (topology) --> original work does not generate node features (label included)

graph sample from p(data) distribution --> learn the distribution --> generate new graph
 

### key point
- maximu likelihood --> find the model that is most likely to have generated the observed data

### sample
- sample from noise distribution --> transform the noise via function
    --> how to design the function? --> use deep neural networks
- start with random seed and then expand for generating the graph

### auto-regressive model
- same neural network used for both density estimation and sampling --> other approaches like GANs have 2 or more models for doing the generation
- generate a graph as step-by step action
- represent a graph as  asequence, a set of actions --> probability of next action is based on the previous actions

### generating realistic graphs
- sequentially adding nodes and edges
- we need node ordering --> we can create a sequence for generating the graph --> sequence uniquely defined
    --> for each node I ask "should I link to node X?" --> I make this question for each node --> it becomes a sequence problem --> each step outputs a probability of a single edge

2 level sequence, for each node step, I do an edge level step
- sequence of adding nodes
- sequence of adding edges

### problem
- transformed grah generation problem into a sequence generation problem
- we have 2 process:
 1) node generation process
 2) edge generation process
- we can use Recurrent Neural Networks because we have a sequence of actions

### The RNN
- RNN sequentially takes input sequence to update its hidden states --> hidden state summarize all the infromation input to RNN
- hidden state of previous step is updated weighted and used as hidden state of the next step
- output of RNN is the prediction scalars if the node is present or not --> Bernoulli distribution TO CHECK
- SOS (start of sequence ) as initial input
- EOS (End of sequence) as final output and stop generation when produced (in this case based on the output of the RNN edges)


### Training
- teacher forcing --> technique of features forcing --> replace input and output with real sequence TO CHECK
    --> instead of using predicted output as input for the next step, we use the real output --> teacher forcing the student --> correct the output of each step to avoind having the issue to all the steps
- Binary Cross Entropy (TO CHECK) --> loss function --> will adjust RNN parameters accordingly using back propagation


### Scaling Up and evaluation of graph generation
- we limit complety with max_prev_nodes --> otherwise last added node will check if should be connected also to the first node --> we limit the check to the last X nodes
- breadth first search (TO CHECK) --> for ordering the nodes
- similarity by visualization
- statistical measure for graph comparison (TO CHECK) --> 
    1) Earth Mover Distance (EMD) --> measure discrepancy between 2 distributions
    2) Maximum Mean Discrepancy (MMD) based on EMD --> distance between mean embeddings of features


### Future Work
- Graph Convolutional Policy Network (GCPN) 
