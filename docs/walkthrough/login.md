(login-to-jina-ecosystem)=
# Login

Since Finetuner leverages cloud resources for fine-tuning,
you are required to {meth}`~finetuner.login()` (or `~finetuner.notebook_login()`) and obtain a token from Jina before starting a fine-tuning job.
It is as simple as:

```python
import finetuner

finetuner.login()  # use finetuner.notebook_login() in Jupyter notebook or Google Colab
```

A browser window should pop up with different login options.
After {meth}`~finetuner.login()` or `~finetuner.notebook_login()` you will see this in your terminal:

```bash
🔐 Successfully logged in to Jina AI as [USER NAME]!
```

 Now, an authentication token is generated which can be read with the {func}:`~finetuner.get_token` function.
If you have been logged in before, the existing token will not be overwritten, however, if you want this to be happen, you can set the `force` attribute in the login function to true.

```
finetuner.login(force=True)
```

```{admonition} Why do I need to login?
:class: hint
Login is required since Finetuner needs to push your {class}`~docarray.array.document.DocumentArray` into the Jina AI Cloud as training data.
Once you have successfully logged in, your training data will be linked to your personal user profile and will only be visible to you.

Once fine-tuning is completed, the fine-tuned model will be visible only to you in the Jina AI Cloud.
```