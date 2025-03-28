# Setting Up SSH for GitHub

## Videos on youtube:
- https://www.youtube.com/watch?v=iVJesFfzDGs
- https://www.youtube.com/watch?v=X40b9x9BFGo

## **Why Use SSH?**
Previously, you used HTTPS to interact with GitHub. SSH provides a more secure and convenient way to authenticate without repeatedly entering your credentials.

## Steps to setup SSH for GitHub

### **Step 1: Check for Existing SSH Keys**

Open a terminal (on Windows, use Git Bash) and check if you already have an SSH key:
```bash
ls -al ~/.ssh
```

- If you see files like `id_rsa.pub` or `id_ed25519.pub`, you already have an SSH key. Go to Step 3.
- If you see nothing, or if you see an error (e.e.g no .ssh folder, cannot access folder), go to Step 2.


### **Step 2: Generate a New SSH Key (If Needed)**
If you don't have an SSH key, generate one using:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept the default file location and optionally add a passphrase.

Most likely, you need just to tape "Enter" key three times, so that no passphrase is needed: it will save you a lot of time.

### **Step 3: Add the SSH Key to the SSH Agent [Optional]** 
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### **Step 4: Add the SSH Key to Your GitHub Account**
Copy your SSH public key:
```bash
cat ~/.ssh/id_ed25519.pub
```
Go to GitHub -> **Settings** -> **SSH and GPG keys** -> **New SSH key**, paste your key, and save.

### **Step 5: Test the Connection**
```bash
ssh -T git@github.com
```
If successful, you'll see a message like:
```
Hi username! You've successfully authenticated.
```
Now, you can use SSH for Git operations.

If you see things like `github doesn't provide shell access`, there is no problem.


## Test Cloning GitHub with SSH

```
git clone git@github.com:evidentiallab/Lunde_Chen_NN_Belief_Reproduction.git
```

You should be able to see the cloning goes smoothly.


## Change your remote url to use SSH [Optional]

For your previous repositories, you can change the remote url to use SSH by running:

```bash
git remote set-url origin git@github.com:yourusername/your-repo.git
```


