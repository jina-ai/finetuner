(login)=
# {octicon}`sign-in` Login

Finetuner, being a product of [Jina](https://jina.ai/) ,
uses [Jina AI Cloud](https://cloud.jina.ai/) for authentication.

Accessing Jina AI Cloud is straightforward, and you can do so by following these steps:

```python
import finetuner

finetuner.login()
```

Once you execute the `~finetuner.login()` method,
a browser window will appear, offering various login options.
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

By default, the data and fine-tuned model are only visible to you, unless you choose to manually set them to public.
```

In case the login process fails,
you can try logging in again by using the `force` option to force re-login.

```python
import finetuner

finetuner.login(force=True)
```
