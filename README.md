

# Exam template Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked to.

For both functions to work you mustn't rename anything. The script has two dependencies that can be installed with

```bash
pip install click markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the GitHub repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the GitHub repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

### Question 2
> **Enter the study number for each member in the group**
>
> Anjali Sarawgi            12690415
> 
> Ali Najibpour Nashi       12644070
> 
> Annas Namouchi            12845130
> 
> John-Pierre Weideman      12696407

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**

Answer:

> 1.  PyTorch: Provided tools for building and training our deep learning models, including nn.Module for defining neural network architecture and torch.optim for optimization algorithms.
> 2.  Conda: Managed dependencies and environments.
> 3.  Cookiecutter: Helped us set up our initial project structure quickly with best practices.
> 4.	Hydra: Used for configuration management, to improve workflow and reproducibility.
> 5.	WandB: Experiment tracking and visualization, crucial for monitoring training progress and hyperparameter tuning.
> 6.	Docker: Ensured consistent environments across development, testing, and production by containerizing our application.
> 7.	Pytest: Used for writing and running tests, ensuring code reliability and correctness.
> 8.	GitHub Actions: Automated our CI/CD pipeline, running tests and deploying applications on code commits.
> 9.  PEP8: Ensured our code adhered to Python’s style guidelines.
> 10.	Cloud Services: Used cloud infrastructure for scalable and efficient model training and deployment.
> 11.	FastAPI: Provided a way to build and deploy our API for model inference.


## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
>
> Answer:
> 
> We used a virtual environment to isolate the project's dependencies from the system-wide Python packages.
> 
> We kept track of all the packages and their versions in a requirements.txt file. This file lists all the dependencies required by the project, and a new team member will have to install these.
> 
> We also created a Dockerfile to containerize the application. This ensures that the project runs in the same environment regardless of where it is deployed. The Dockerfile includes all the necessary steps to set up the environment, including installing dependencies. Thus instead of using the requirements.txt file for the environment they can instead build and run the Docker container.



### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**

> Answer:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Did not fill
│   │
│   ├── index.md         
│   │
│   ├── mkdocs.yml      
│   │
│   └── source/          
│
├── models               <- .pth files of the model weigths that we obtain from training
│
├── notebooks            <- Jupyter notebooks. We didn't fill this folder
│
├── pyproject.toml       <- Project configuration file
│
├── reports              
│   └── figures          <- Generated graphics and figures from our training and prediction results
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_sign_mnist     <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py     <- profiling, wandb, autopep8, typing, docker incomplete, dvc check
│   │   ├── scripted_model.pt     <- api testing and attempted local deployment 
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations. We didn't fill this folder
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer:
> 
> PEP8 Compliance: We ensured that all our Python code adheres to the PEP8 style guide, which includes guidelines for code layout, naming conventions etc.
> 
> In big projects, this is important because it ensures:
> 
> Maintainability: Consistent code is easier to read, understand, and modify.
> 
> Collaboration: Standardized code facilitates teamwork and reduces onboarding time for new members.
> 
> Error Prevention: Style guides and type hints help catch errors early.
> 
> Documentation: Clear documentation aids in understanding and future maintenance.
> 
> Automation: CI/CD pipelines ensure consistent code quality and catch issues early.
> 
> Scalability: High-quality code is easier to scale, refactor, and extend with new features.
> 


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer:
> 
> In total we implemented 2 tests
> 
> test_api.py:
> 
> - Tests API Endpoints: Ensures endpoints are accessible.
>   
> - Validates Responses: Checks that API responses have the correct data structure and content.
>   
> - Handles Errors: Verifies API behavior for invalid inputs and missing parameters.
>   
> test_data.py:
> 
> - Data Processing Tests: Verifies if data files exists, we have the correct number of samples, and if all our labels are present.

>   
> test_model.py:
> 
> - Model Tests: we test if the model archiutecture runs smoothly given a dummy input.
> 


### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
> 
> Answer:
> 
> The total code coverage of our code is 59%, which includes all our source code. We are far from 100% coverage, and even if we achieved 100%, it would not guarantee that the code is error-free.
> 
> Code coverage measures how much of the code is executed during testing, but it does not assess the quality of the tests themselves.
> 
> There can still be issues that are not caught by tests.
> 
> Code coverage also does not account for the correctness of the output or the handling of unexpected inputs.



### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer:
>
> We made use of both branches and pull requests in our project. In our group, members had a branch that they worked on in addition to the main branch.
> 
> Each feature or bug fix was developed in a separate branch, which allowed us to work on multiple parts of the project individually, but simultaneously without interfering with the main codebase.
> 
> To merge code, we would create a pull request from our individual feature branch to the main branch. This triggered a review process where other team members could review the changes, run tests, and provide feedback.
> 
> Only after the PR was reviewed and approved, the changes were merged into the main branch.
> 
> Using branches and PRs also allowed us to maintain a clean and stable main branch.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer:
> 
> We did used DVC for managing data in our project.
> 
> It allowed us to track changes to our datasets, ensuring that we could reproduce experiments and results.
> 
> DVC enables us to maintain different versions of our datasets, making it easy to switch between different stages of data processing and analysis.
> 
> With DVC, we could share data easily, without having to manually manage large files or deal with inconsistencies.
> 
> DVC's integration with cloud also helped us keep our data accessible.


### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Answer:
> 
> We have organized our continuous integration (CI) setup using GitHub Actions
> 
> Our CI setup includes:
> 
> 1. **Unit Testing:** We run unit tests using `pytest` to ensure that our code behaves as expected. This is done automatically on every push to the repository with help from GitHub Actions.
>    
> 2. **PEP8 compliance:** We use `ruff` to enforce PEP8 compliance and catch syntax errors or style issues.
>  
> 3. **Testing Multiple Python Versions:** Our CI setup includes testing across multiple Python versions 3.9, 3.10 and 3.11 to ensure compatibility and identify any version-specific issues.
>    
> 4. **Caching:** We use caching to speed up the CI process by storing dependencies between builds. This reduces the time taken to install dependencies on each run.


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer:
> 
> We configured our experiments using config files with Hydra. This allowed us to manage and modify hyperparameters.
> 
> Here is an example of how we set up and ran an experiment:

```txt
 hydra:
  run:
    dir: .
  output_subdir: null
  job_logging:
    level: INFO

 hyperparameters:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 3
 ```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:
> We made use of config files, version control, experiment logging, data versioning, and Docker to ensure reproducibility of our experiments. Whenever an experiment is run, the following happens:
> 
> To reproduce an experiment, one would need to:
> 
> 1. Checkout the correct Git commit for the code and configuration.
> 2. Use DVC to pull the corresponding data version.
> 3. Build and run the Docker container to ensure the environment is consistent.
> 4. Run the experiment script with Hydra to apply the saved configuration.
> 
> This is how we guarantee reproducibility.
"""
--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:


--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:
> For our project, we developed Docker images for training and prediciton of our model.
> To run the training Docker image, we used the following command:
> **docker run trainer:latest --config config.yaml**
> 
> Here is a link to our Docker files:
> [mlops_sign_mnist/dockerfiles/predict_model.dockerfile](https://github.com/anjalisarawgi/mlops_sign_mnist/blob/main/mlops_sign_mnist/dockerfiles/predict_model.dockerfile)
> [mlops_sign_mnist/dockerfiles/train_model.dockerfile](https://github.com/anjalisarawgi/mlops_sign_mnist/blob/main/mlops_sign_mnist/dockerfiles/train_model.dockerfile)

"""
--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:
> When running into bugs while trying to run our experiments, we performed debugging using a combination of logging, interactive debugging tools, and unit tests.
> 1. Logging: We used  logging to track the flow of execution and identify where the code was failing. By examining the logs, we could pinpoint the source of errors.
> 2. Interactive Debugging: The VSCode IDE which allowed us to use debugger.
> 3. Print statements: We also used print statements for debugging
> 4. Unit Tests: We used unit tests to help us verify that individual components of our code were functioning correctly.
>
> We think our code is perfect

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:
> In our project, we used the following Google Cloud Platform (GCP) services:
> Google Cloud Storage: Used for storing and managing our data files.
> Google Cloud Functions: Used for deploying serverless functions.
> Google Cloud Run: Deployed our containerized applications in a fully managed environment
> Google Cloud Build: Automated the building and pushing of our Docker images, ensuring that our containers were built correctly every time we pushed changes to github.
> Google Compute Engine: Provided virtual machines for running various workloads.
> Google Cloud Buckets: Part of GCS, used for organizing and storing our data files and model weights. 
>

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:
> We used the Google Compute Engine to run our machine learning training and inference workloads. Compute Engine provided us with virtual machines (VMs) for our computational tasks.
> 
> We used with the following instance:
> - **n1-standard-1:**
> - **region: europe-west1-b**
> We started this instance using a ready to go Pytorch image (pytorch-latest-cpu-v20240627) to ensure consistency across different environments.
> We cloned a github repository in this instance, set up the environment and ran our training model.
> Our VM is **mlops-sign-mnist-instance**


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP artifact registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:
> We managed to deploy our model locally and in the cloud. For deployment, we used a Fast API application.
> Initially, we served the model locally to ensure that the deployment process worked correctly. We used Docker to containerize the application.
> After local testing, we deployed the Docker container to Cloud.
> To invoke the deployed service, users can send a POST request to the API endpoint with the input data. Here is an example of how to call the service:
> 
> curl -X POST -H "Content-Type: application/json" -d '{"data": [1, 2, 3, 4]}' <weburl>
> 
> This allows users to easily interact with our model.
"""
--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:
> 
> We did not implement monitoring. We would could implemented monitoring so that over time we could measure performance metrics and detect any anomalies in our model's behavior.
> Monitoring would help us track the model's performance, such as response times, error rates, and resource usage. This information would inform us about the health and efficiency of our application.
> Additionally, monitoring could provide insights into data drift, where the input data distribution changes over time, potentially affecting model accuracy. By implementing monitoring, we could proactively address performance issues and ensure the longevity and reliability of our deployed model.
>
--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Recommended answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:
> In total we used $2.86 of credits during development. The service costing the most was Compute Engine due to computational power required to handle large datasets.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
> First we setup our project locally. We use tools such as cookiecutter, Pytorch, Conda, Git and DVC
> The overall steps:
> Local Machine:
> - Experiment Configuration and Submission: Experiments are configured using Hydra and tracked using W&B. This local setup allows us to use hyperparameters from config files.
> - GitHub (Version Control):
> - Source Code Management: We commit and push our changes to a shared GitHub repository.
> - We run our project locally using Docker.
> Google Cloud:
> - Model Training: The data and model weights are pushed to Cloud for further training. We also use Compute Engine to train our model on cloud.
> GitHub Actions (CI/CD):
> - Continuous Integration and Testing: Upon code submission, GitHub Actions is triggered to run automated tests, including unit tests, linting (ruff).              
> Google Cloud Platform (GCP):
> - Docker Image Build: Changes pushed to GitHub trigger a build in GCP, creating a new Docker image with the latest code. This image is stored in Google Container Registry.
> - Deployment: We create a docker image and push it to CLoud for our Fast API application. We then run the deployment on cloud.
> User Interaction:
> - Code and Image Access: Users can clone the source code from GitHub and pull the newest Docker image to ensure they are working with the latest version of the application. Additionally they can also use our API  to make predictions.

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:
> The biggest struggles in the project were related to integrating with the cloud and developing the API.
> Cloud Integration:
> - One major challenge was configuring and managing the various GCP services required for our project. Setting up the Google AI Platform for training, managing data with Google Cloud Storage, and ensuring interaction between services was frustrating and time-consuming.
> - We also faced difficulties with setting up permissions and ensuring secure access to our cloud resources. To overcome these challenges, we dedicated time to thoroughly read GCP documentation, followed step-by-step tutorials
> API Development:
> - Developing an API for model inference was nother challenge.
> - We encountered issues with request handling and ensuring the API could process and return predictions. To address these, we implemented logging and monitoring to track API performance and debug issues.
> Time Management:
> - A considerable amount of time was spent debugging issues related to cloud services and the API. Regular team meetings also helped us stay on track and share our knowledge, and work on problems together.

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
> John-Pierre Weideman was in charge of setting up the Cookiecutter project structure and developing the Docker containers for training our applications. He also worked on writing the initial data processing scripts and setting up version control with GitHub and DVC.
> Ali Najibpour Nashi focused on creating the training scripts and setting up continuous integration with GitHub Actions. He also implementing the initial unit tests and ensured that our code is PEP8.
> Anjali Sarawgi trained our models in the cloud using Google Cloud and managed the deployment process. She also handled the integration of Weights & Biases for experiment tracking and monitoring.
> Annas Namouchi was responsible for developing and deploying the API for model inference. Additionally, he also worked on profiling and setting up config files using Hydra.
