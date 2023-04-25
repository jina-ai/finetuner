---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="p8jc8EyfruKw" -->
# Getting Started (Image-to-Image Search with TripletMarginLoss)

<a href="https://colab.research.google.com/drive/1jg9KiAzhhokYctA0wc0hOSO4RumIwa6_#scrollTo=p8jc8EyfruKw"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Welcome to Finetuner's "Getting Started" guide! This tutorial will walk you through the process of fine-tuning a model for a neural search application, specifically focusing on an image-to-image retrieval task using the [UKBench dataset](https://archive.org/details/ukbench). The figure below illustrates the typical workflow when working with Finetuner:


<!-- #endregion -->

<!-- #region id="NL4dU4Jqytyj" -->
![](https://user-images.githubusercontent.com/6599259/233098291-42a0bf8b-6a7c-4da9-b319-13f61ee83171.png)
<!-- #endregion -->

<!-- #region id="O0OPJ59l1X0-" -->
In particular, you need to perform the following task for our image-to-image search example:

**1. Data Preparation:** Parse the training and evaluation datasets from the *UKBench dataset* and convert them into a specifically formatted CSV file or a [DocArray](https://github.com/docarray/docarray) dataset, which is Finetuner's internal format.

**2. Configure and Submit the Fine-Tuning Job:** Create a fine-tuning job by configuring the desired model and its training parameters. After executing the submission (fit) function, Finetuner initiates a cloud computing job that carries out the training on high-end GPU machines.

**3. Monitor your Job:** Fine-tuning can be time-consuming. While it is in progress, you can monitor the cloud job (retrieve logs, track evaluation metrics, etc.) using either a Python script or [Finetuner's Web interface](https://cloud.jina.ai/user/finetuner).

**4. Use the Model:** Once the fine-tuning is complete, you can download the model for local use to embed queries and documents or deploy a microservice that performs the encoding and seamlessly integrates with a search application.

---
So far, so good, now detailed explainations of those steps follow. 

*Note, please switch to a GPU/TPU Runtime in the Colab setting or the local encoding at the end of the tutorial will be extremely slow!*
<!-- #endregion -->

<!-- #region id="MjRR_mPVykAh" -->

## Install

First of all, you need to install `finetuner`. Installing the "full" package also includes all the dependencies you need to use the models locally. In addition, `jina` is installed for deploying a model in an encoding service.
<!-- #endregion -->

```python id="VdKH0S0FrwS3"
!pip install 'finetuner[full]'
!pip install jina
```

<!-- #region id="M1sii3xdtD2y" -->
## Data Preparation

You can start with investigating the retrieval problem. The images of the UKBench datasets can be viewed on this website:

https://ia600809.us.archive.org/view_archive.php?archive=/5/items/ukbench/ukbench.zip
<!-- #endregion -->

<!-- #region id="UFLeYrKd_odN" -->
Before continuing with the implementation, you can take a look at data! The dataset is supposed to be used to evaluated near duplicate detection. It includes images of scenes and objects like a variety of CD covers. Every object is depicted by four images. To better process process the images, the following code creates a DocumentArray object of the image urls. Then the `plot_image_sprites()` can display them.
<div>
<img src="https://user-images.githubusercontent.com/6599259/233361492-fbf05c48-5b3a-49d7-bc92-32e4a0aad4f7.png" width="400"/>
</div>
You can try it out yourself by running it in the notebook: 
<!-- #endregion -->

```python id="JkHvkJ18kxLK"
from docarray import Document, DocumentArray

DATASET_BASE_URL = (
    'https://archive.org/download/ukbench/ukbench.zip/full%2Fukbench{fname}.jpg'
)

dataset = DocumentArray()
for number in range(2000):
    image_doc = Document(
        uri=DATASET_BASE_URL.format(fname='{:05d}'.format(number)),
        tags={'number': number},
    )
    dataset.append(image_doc)

dataset[:8].plot_image_sprites()
```

<!-- #region id="XOTogw82Epkv" -->
**The task:** This tutorial considers the following retrieval problem: The inputs are one of the four image files of an object, and the goal is to retrieve the corresponding three duplicates - images of the same object.

**How to search:** To accomplish this, you can fine-tune an embedding model for encoding images into embedding respresentations. Image files diplaying the same object should be assigned to similar vector representations after the training. Then, you can use the model to implement a search system which encodes an input (query) image with the model and determines in a collection of image embedding representations encoded in advanced the nearest neighbors to select similar pictures.

**Construct the datasets:** To prepare the data for the model, you'll split the dataset into a training and a hold-out set for testing. You will then transform the hold-out set into a set of queries and an image collection.


<!-- #endregion -->

```python id="r1npa-IDnf3G"
from docarray import Document, DocumentArray

train_files, test_files = dataset[:1_000], dataset[1_000:]
```

<!-- #region id="wZHIlCcy4Fq9" -->
For training, you need to store the files in a `DocumentArray` object where each file is assigned to a label defined by the `'finetuner_label'` key in the tags attribute. Since the images are sorted by the objects in the dataset, we can simply defined the labels by considering the positions of the images in our dataset. Four consecutive images are assinged to the same label by dividing the position by four and round it down. 

To load the iamges into main memory, you can apply the `load_uri_to_blob` function. This is especially important if you create datasets with images stored in your local file system which can otherwise not be loaded by the cloud fine-tuning job. Besides, it reduces the runtime of the clould job.

As an alternative to creating a DocArray dataset, you can prepare a CSV file of the filenames and labels as described in our [documentation](https://finetuner.jina.ai/walkthrough/create-training-data/).
<!-- #endregion -->

```python id="7T97j9DRnv8f"
from tqdm import tqdm
from docarray import Document, DocumentArray

def create_training_dataset(image_dataset):
  train_dataset = DocumentArray()
  for i, image_doc in tqdm(enumerate(image_dataset), total=len(image_dataset), desc='Pre-process training data'):
    train_doc = Document(uri=image_doc.uri, tags={'finetuner_label': (i // 4)}).load_uri_to_blob()
    train_dataset.append(train_doc)
  return train_dataset

train_data = create_training_dataset(train_files)
```

<!-- #region id="p4co8IHa76ms" -->
In the training dataset, queries and documents are not distinguished. However, to calculate retrieval metrics like *Recall*, *Precision*, or *MRR*, it is necessary to explicitly define a set of queries and an index dataset of images from which to select results. Thus, the following code iterates through the test set and treats the first of four images as queries and the remaining images as an index document collection. Again, the file names represent the labels to judge the relevancy of images to each other.
<!-- #endregion -->

```python id="yqFrp-hPxYTB"
query_data = DocumentArray()
index_data = DocumentArray()
for i, image_doc in tqdm(enumerate(test_files), total=len(test_files), desc='Pre-process evaluation data'):
  if i % 4 == 0:
    query_data.append(Document(uri=image_doc.uri, tags={'finetuner_label': (i // 4)}).load_uri_to_blob())
  else:
    index_data.append(Document(uri=image_doc.uri, tags={'finetuner_label': (i // 4)}).load_uri_to_blob())
```

<!-- #region id="7EliQdGCsdL0" -->
## Configure and Submit your Finetuning Job

Now, you can submit the fine-tuning job. The pre-trained model used in this tutorial is the popular ResNet50 model. It is based on convolutional layers and is good at solving computer vision problems. The ResNet model used here is pre-trained on the [ImageNet]((https://www.image-net.org/)) classification task. Finetuner will remove the classification head of the model so that it outputs embedding vectors instead.

To submit the job, you first need to log in to Jina:
1. Execute the `login` function
2. Click on "Browser Login"
3. Click on the login link
4. Log in with your preferred login method
<!-- #endregion -->

```python id="NIXc9Jyih_JD"
import finetuner
finetuner.login(force=True)
```

<!-- #region id="kR5dTgITh_vp" -->
After that, you can configure everything else in the fit function, which submits the job to the cloud.

The configuration includes:
- The name of the model: `resnet50`
- The training data object
- The loss function: `TripletMarginLoss` works well for fine-tuning embedding models with lots of different labels - for instance lots of queries with only a few matches as in the Totally Looks Like dataset.
- Hyperparameters: For ResNet50, the following works good in general: batch_size=128, epochs=5, and learning rate = 1e-4
- An `EvaluationCallback`: It assesses the retrieval performance during the training on the query and index dataset. The `limit=3` attribute tells the callback to only considers the 3 top ranked results for each query.

The `fit` function returns a run object, which we can use to monitor the run in the next step.
<!-- #endregion -->

```python id="qGrHfz-2kVC7"
import finetuner
from finetuner.callback import EvaluationCallback

run = finetuner.fit(
    model='resnet50',
    train_data=train_data,
    loss='TripletMarginLoss',
    batch_size=128,
    epochs=5,
    learning_rate=1e-4,
    callbacks=[
        EvaluationCallback(
            query_data=query_data,
            index_data=index_data,
            limit=3,
        )
    ],
)
```

<!-- #region id="7ftSOH_olcak" -->
## Monitor your Job

Fine-tuning can take a while (~15min). After submitting your job, Finetuner starts an instance to run it, downloads your training data, and begins the training process. To check the current status, call `run.status()`. After the training has started, you can use `run.logs()` to retrieve the logs. To continuously print the log messages to the console, you can use the `run.stream_logs()` function.

The following code snippet waits for the job to start and then streams the logs to the console:
<!-- #endregion -->

```python id="xM9fLTyqcUEJ"
for message in run.stream_logs():
  print(message)
```

<!-- #region id="kQmpEuGteGtL" -->
Following the logs, you can see the steps performed by Finetuner to train the model. Moreover, it displays evaluation metrics determined by the callback after each epoch. As you can see, there is a significant improvement.
<!-- #endregion -->

<!-- #region id="-RhURjxIiPAO" -->
### Montior in Jina AI CLoud UI

As an alternative to the `finetuner` package, you can also use the Web UI at https://cloud.jina.ai/ for monitoring your run. After logging in, click on the "Finetuner" section in the left sidebar to view your runs and their statuses. If you don't know your run's name, check the `run.name` property. When you click on one specific run, a view with more details about it appears. For instance, you can view the logs of the run there.

<!-- #endregion -->

<!-- #region id="cQem9kQ7RUYZ" -->
<div>
<img src="https://user-images.githubusercontent.com/6599259/233099591-d27405b3-a26c-4951-81df-2c5dc096113e.png" style="float: left" width="500"/>
<img src="https://user-images.githubusercontent.com/6599259/233099603-6af406e1-15c1-401b-af5a-495404114f4c.png" width="500"/>
</div>
<!-- #endregion -->

<!-- #region id="geAbdSPzisiN" -->
## Download the Model and Encode Documents

After the code block above printed the last message to the console, `run.stream_logs()` terminates, and `run.status()` would return `FINISHED`. Now, you can download the model for local use with a Python script, or set up an encoding service on your local machine or in Jina AI Cloud. The latter is the recommended method for using fine-tuned models in production.

To download the model, call the `finetuner.get_model` function with the artifact ID of the model stored in the `run` object. Afterward, you can use the `finetuner.encode` function to encode images from the evaluation set.


<!-- #endregion -->

```python id="ZJLGNd67iM-G"
import finetuner

IMAGE_URI = 'https://archive.org/download/ukbench/ukbench.zip/full%2Fukbench00000.jpg'

model = finetuner.get_model(run.artifact_id)
embeddings = finetuner.encode(model=model, data=[IMAGE_URI])
print(embeddings)
```

<!-- #region id="IICjO5gLbEU2" -->
To investigate how fine-tuning changed the results, you can also build the pre-trained model with the  `finetuner.build_model` function and use DocArray's `match` function after the encoding to obtain the Top-K results before and after fine-tuning

(The code below only selects queries where the results between both models differ):
<!-- #endregion -->

```python id="NVaE8lZzcDsO"
pretrained_model = finetuner.build_model('resnet50')

# encode queries and index data with the pre-trained model
pretrained_queries = finetuner.encode(
    model=pretrained_model, data=DocumentArray(query_data, copy=True)
)
pretrained_index = finetuner.encode(
    model=pretrained_model, data=DocumentArray(index_data, copy=True)
)

# encode queries and index data with fine-tuned models
finetuned_queries = finetuner.encode(
    model=model, data=DocumentArray(query_data, copy=True)
)
finetuned_index = finetuner.encode(
    model=model, data=DocumentArray(index_data, copy=True)
)

# matching
pretrained_queries.match(pretrained_index)
finetuned_queries.match(finetuned_index)

# show some results
shown_results = 0
for i in range(len(pretrained_queries)):
    if shown_results > 4:
        break
    if not all(
        [
            pretrained_queries[i].matches[j].tags['finetuner_label']
            == finetuned_queries[i].matches[j].tags['finetuner_label']
            for j in range(3)
        ]
    ):
        print('pretrained:')
        pretrained_queries[i].plot_matches_sprites()
        print('finetuned:')
        finetuned_queries[i].plot_matches_sprites()
        shown_results += 1
```

<!-- #region id="Nt2VDVkniKnH" -->
You can retrieve the metrics of this optimized run by using the `display_metrics` function:
<!-- #endregion -->

```python id="lq2eGCnziQ0n"
run.display_metrics()
```

<!-- #region id="CnZDKagtoddx" -->
```bash
Retrieval metrics before fine-tuning:              
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Retrieval Metric           ┃ Value              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ resnet50_precision_at_k    │ 0.936              │
│ resnet50_recall_at_k       │ 0.936              │
│ resnet50_f1_score_at_k     │ 0.936              │
│ resnet50_hit_at_k          │ 0.992              │
│ resnet50_average_precision │ 0.9826666666666665 │
│ resnet50_reciprocal_rank   │ 0.9853333333333333 │
│ resnet50_dcg_at_k          │ 2.486170745114311  │
└────────────────────────────┴────────────────────┘
Retrieval metrics after fine-tuning:               
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Retrieval Metric           ┃ Value              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ resnet50_precision_at_k    │ 0.9533333333333333 │
│ resnet50_recall_at_k       │ 0.9533333333333333 │
│ resnet50_f1_score_at_k     │ 0.9533333333333333 │
│ resnet50_hit_at_k          │ 1.0                │
│ resnet50_average_precision │ 0.9943333333333333 │
│ resnet50_reciprocal_rank   │ 0.996              │
│ resnet50_dcg_at_k          │ 2.5293130592000264 │
└────────────────────────────┴────────────────────┘
```
<!-- #endregion -->

<!-- #region id="EbVD9FHjjOIg" -->
## Set up an Encoding Service

If you want to use the model in production, you might want to set up a microservice that does the encoding. For this purpose, Jina provides the [FinetunerExecutor](https://cloud.jina.ai/executor/13dzxycc?random=4209), which can easily integrate fine-tuned models with the Jina ecosystem.

The [jina package](https://github.com/jina-ai/jina) allows you to run a service that uses the FinetunerExecuter locally. You need to configure the artifact id of your model and an authentication token, which the service requires to download your fine-tuned model. You can then start it and send an DocumentArray object with an image to the service to get its embedding, as you can see below: 
<!-- #endregion -->

```python id="UaqBJfj7ysKZ"
import finetuner
from docarray import Document, DocumentArray
from jina import Flow

IMAGE_URI = 'https://archive.org/download/ukbench/ukbench.zip/full%2Fukbench00000.jpg'

token = finetuner.get_token()

f = Flow().add(
    uses='jinaai://finetuner/FinetunerExecutor:latest',
    uses_with={'artifact': run.artifact_id, 'token': token}
)

with f:
  returned_docs = f.post(
        on='/encode',
        inputs=DocumentArray(
            [
                Document(
                    uri=IMAGE_URI
                )
            ]
        )
    )
print(returned_docs[0].embedding)
```

<!-- #region id="AsrIZHe6RffZ" -->
Nevertheless, hosting a service on your local machine might not be practical for a production use case. Instead, you can host a service in the Jina AI Cloud. For this purpose, you need to create a YML config to describe your Flow. This config references the FinetunerExecutor and contains the artifact id and the authentication token, similar to the in-code configuration in the example above. The following code block creates such a config file and stores it in the current directory:
<!-- #endregion -->

```python id="84O4ynm_Ap6h"

from jina import Flow

token = finetuner.get_token()

yaml_file = f"""jtype: Flow
executors:
  - uses: jinaai+docker://finetuner/FinetunerExecutor:latest
    timeout_ready: -1
    install_requirements: true
    uses_with:
      artifact: {run.artifact_id}
      token: {token}
"""

with open('flow.yml', 'w') as f:
  f.write(yaml_file)

```

<!-- #region id="_HPkw_aaR7O6" -->
Now, you can host the service via the Jina Cloud CLI (JCloud):
<!-- #endregion -->

```python id="sMmHmOgLCd00"
!jcloud deploy flow.yml
```

<!-- #region id="b_SAjOb3SGmb" -->
Usually, this prints the host URL of the endpoints where you can send documents to the console. However, in the notebook, it might not show up. In this case, one can use the following snipped to retrieve the host URL and store it in the host variable (If you have other flows running, this might not work - please set the host variable manually):
<!-- #endregion -->

```python id="HpH8mkzgPHLH"
from jcloud.flow import CloudFlow

x = (await CloudFlow().list_all(phase='Serving'))['flows']
host = x[0]['status']['endpoints']['gateway']
print(host)
```

<!-- #region id="TUoAAJALdZ5r" -->
Alternatively, you can go to https://cloud.jina.ai/user/flows which should list your Flow:
![image.png](https://user-images.githubusercontent.com/6599259/233099611-db796a67-8493-4250-9c31-e0f58c0ce2c7.png)
<!-- #endregion -->

<!-- #region id="ueSu_ih2SmlP" -->
Now, you can initialize a client and use it to send an image to the service to encode it. Since the service can not access your local file system, it is necessary to load the image into memory before sending it. You can do this with DocArray's `load_uri_to_blob()` method to load images as BLOBs into the documents:
<!-- #endregion -->

```python id="7qQQ2RwUTFG8"
from jina import Client

client = Client(host=host)

documents = DocumentArray([Document(uri=IMAGE_URI).load_uri_to_blob()])

response = client.post('/encode', documents)

print(response[0].embedding)
```

<!-- #region id="c2E7BZniU79o" -->
Finally, you should terminate the service either in the Web UI or by using the CLI. The following command shut down all of your currently running flows:
<!-- #endregion -->

```python id="37k40tSoVDKE"
!jcloud remove all
```

<!-- #region id="M_x0Eg-MZsfw" -->
## Conclusion
In this guide, we walked through the process of fine-tuning an image-to-image retrieval model using Finetuner. We covered data preparation, configuring and submitting a fine-tuning job, monitoring the job, and using the fine-tuned model both locally and in a microservice.

As you've seen, Finetuner makes it easy to fine-tune pre-trained models for various tasks. By fine-tuning, you can significantly improve retrieval performance, ensuring more accurate and relevant results for your search applications. Keep in mind that hyperparameter optimization is crucial to achieve the best possible results, but it may come at the cost of increased training time.

<!-- #endregion -->
