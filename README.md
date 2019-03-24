# Content of the repository

* `adversarial_examples.md`: Some references to papers about adversarial examples.
* `model_stealing_logistic_regression.ipynb`: Example how to steal a logistic regression model.
* `adversarial_example.ipynb`: Example how to create an adversarial example.

# Setup a virtual environment to run the Jupyter demos

    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install tensorflow keras jupyter matplotlib torch torchvision tqdm