# Push the Cybertron project into the existing repository (Mac terminal)

These are **real commits made by you now**, not artificially backdated history.

Prerequisites: Git installed (`git --version`); your own GitHub account has permission to push to the existing repository. If you cannot push, ask Isira-Himal-Perera to add you as a collaborator or let that repository owner perform these commands.

1. Download and extract `Cybertron_A03_EN3150.zip` into your Downloads folder.
2. Open Terminal and paste these commands ONE BY ONE:

```bash
cd ~/Downloads
git clone https://github.com/Isira-Himal-Perera/CNN-for-Edge-Image-Classification.git
cd CNN-for-Edge-Image-Classification
rsync -av --exclude='.DS_Store' --exclude='.git' ../Cybertron_A03_EN3150/ ./
git status
git add .
git commit -m "Add Cybertron CNN models, measured experiments and report"
git push -u origin main
```

Notes:
- If the remote default branch is `master` rather than `main`, replace the final argument with `master`.
- If Git asks you to set your username/email, run `git config user.name 'Your Real Name'` and `git config user.email 'your-verified-github-email@example.com'` with your **actual** data (the displayed address is an example, not a ready-made real email).
- If the repository has commits from your friends, `git clone` keeps that history and `rsync` simply introduces the new files; don't delete the `.git` folder.
- If the password prompt rejects your normal password, authenticate using GitHub CLI (`gh auth login`) or an authorized credential helper/personal-access-token. GitHub doesn't accept account passwords for HTTPS Git pushes.
- If `rsync` says the source folder doesn't exist, check your extracted ZIP folder name and adjust the source path.
- Large `results/models/*.pth` files are intentionally ignored by `.gitignore`; they are included in the downloadable ZIP for reproducibility but are not usually required in GitHub.

Submitting to Moodle: Upload `report/Cybertron_A03_EN3150.pdf` and the source-code ZIP **separately**, as requested by the lecturer. Ensure the dataset spreadsheet entry is completed.
