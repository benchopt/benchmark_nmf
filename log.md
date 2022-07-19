## A few comments on what I found hard/confusion when implementing the benchmark

1. About the general structure

- A diagramm showing how the three main classes rely on each other (e.g. which method calls depends on which other method's input/output). The cross dependency of methods was confusing to me, and I went back and forth writing inputs/outputs in the wrong places.
- Explain what each class should do for these particular cases:
    - Initialization: the runners take care of it; open question: how to share initialization (and not just a random seed?)
    - hyperparameters can belong to methods (e.g. some stepsize), to objectives (e.g. some regularization parameter) or to the data (e.g. some prior knowledge on the rank in NMF). It would be nice to have a clearer distinction between these stated somewhere, with a discussion on how to handle them.
- I just can't seem to make the tests pass :( I tried my best though, not sure what is failing
- I am not sure I like the fact that one run is cut into several calls, which are timed, to build the loss/time curve. Some methods required knowledge of the previous iterations, and cutting the run like this make degrade performances? Why not require the methods to return time? Or at least make sure somehow all the variables are left untouched when chopping the run (maybe this is the case, but it is not clearly explained in the doc?)  


  2. About the runs

- It is still not clear to me how the maximal number of iterations is set. I went deep in benchopt code, and clearly things are happening under the hood to set this maxiter, but I wish it was easier to control the stopping criterion in general? If it is easy, e.g. with the CLI interface, then it is not so clear in the doc.
     (Note: after writing this I found the -n options in the CLI, with default 100; it is not put forth enough)
- I did not try it, but I am not sure how to use the tolerance stopping criterion instead of the maximum number of iterations. Should the methods re-implement the loss and compute it themselves to stop?
- Some instruction for naming the methods would be nice (I realised later that the name matter for calling with the CLI interface)
- Maybe not a bug, but when I print something right after the definition of run, I get a verbose output with chopped first letter, replaced by my print.
 