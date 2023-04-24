(login)=
# {octicon}`sign-in` Login

Finetuner, being a product of [Jina](https://jina.ai/) ,
uses [Jina AI Cloud](https://cloud.jina.ai/) for authentication.

Accessing Jina AI Cloud is straightforward, and you can do so by following these steps:

```python
import finetuner

finetuner.login()
```

Once you call the `~finetuner.login()` method,
a new tab will open in your browser, offering several login options.
After completing the login process, you will receive the following message in your terminal:

```bash
Your browser is going to open the login page.
If this fails please open the following link: ...
🔐 Successfully logged in to Jina AI as [YOUR-USERNAME] (username: your-username)!
```

```{admonition} What happens when I login?
:class: hint
Finetuner will generate a temporary `auth_token` that is associated with your personal account.
This `auth_token` will be utilized by Finetuner to carry out various tasks such as:

1. Sending your training data to [Jina AI Cloud Storage](https://cloud.jina.ai/user/storage).
2. Setting up GPU/CPU resources for performing fine-tuning tasks.
3. Storing fine-tuned models (referred to as Artifacts) on the Jina AI Cloud.
4. Performing billing (applicable only when the user exhausts their free resources).

By default, your data and fine-tuned models are visible only to you unless you publish them by manually setting the `public` flag to `True`.
```

## Namespace

Jina AI Cloud users automatically have a namespace linked to their accounts.
This ensures that even if two users give some artifact or resource the same name,
they can be separately addressed.

If you want to download public data or models published by other users,
you must add `[namespace]/` before the name of the data or artifact.
For example, `finetuner/quora-train-da`,
where `finetuner` is the namespace for the account, and `quora-train-da` is the name of the data,
which refers to the quora training set.

To find the name of your namespace,
log in to [Jina AI Cloud](https://cloud.jina.ai/) and click the `account` button under your profile image.
You will find your namespace in the `Namespace` section.


Step 1             |  Step 2
:-------------------------:|:-------------------------:
![namespace-1](https://user-images.githubusercontent.com/9794489/233982646-9476b885-89a9-45e4-9dd7-eea9127afb4c.jpeg)  |  ![namespace-2](https://user-images.githubusercontent.com/9794489/233982661-25a840a0-6812-4752-96b2-6c692dbf0ead.jpeg)

```{admonition}
Please note that if you are already logged into Jina AI Cloud, you still need to call `finetuner.login()` to initialize the finetuner library correctly with your credentials.
This will join your existing Jina AI Cloud session if one is open. Setting the `force` flag in the login function will create a new session and terminate the previous one.
```

## Troubleshooting

In case the login process fails,
you can try logging in again by using the `force` option to force re-login.

```python
import finetuner

finetuner.login(force=True)
```
