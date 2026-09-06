# BuildStockQuery / AWS Grid-Impact Troubleshooting -- Audit First, Then Iterate

Tamar: paste this whole thing as your first message to Claude (chat or Claude Code, whichever
you have open), then answer its questions as it works through the steps below with you.

---

**Instructions for the assistant helping with this:**

## Ground rules

- Don't assume anything about this setup from what you already know or from any past
  conversation. Treat everything as unverified until it's checked in this session, on this
  machine.
- Plain language only. Define any AWS or Python term the first time it comes up, in a short
  parenthetical.
- Prefer the simplest fix that solves the error in front of you. Don't propose restructuring,
  cleanup, or "while we're at it" changes.
- Work one error at a time. After each fix attempt, ask what happened next -- success or a new
  error -- before proposing anything further. Don't try to predict every downstream error in
  advance.

## Step 0 -- Run the diagnostic first

The project ships a diagnostic that checks the whole AWS setup in the order the model needs it.
Run it from the project folder:

    conda activate cmu-tare-model
    python -m cmu_tare_model.grid_impact.diagnose_bsq_aws

It runs seven checks -- library install, AWS credentials, the Athena workgroup, the query-result
bucket, the Glue tables, BuildStockQuery startup, and one real end-to-end query -- and stops at
the first failure with a specific message about what to change.

Paste the **entire** output, whether it passes or fails.

This replaces most of the manual audit below. Only work through Step 1 by hand for the items the
diagnostic does not cover, or if the diagnostic will not run at all.

Assistant: read `cmu_tare_model/docs/BSQ_AWS_SETUP.md` before diagnosing anything. It explains
what each check means and covers the common failures in its Part 2.

## Step 1 -- Audit what the diagnostic doesn't cover

Gather these and summarize them back in a short plain list:

1. Which version of the project code is here -- run `git log -1 --format="%H %ci"` if this is a
   git repo, otherwise ask when the project folder was last copied or downloaded.
2. The exact `BuildStockQuery(...)` call in the notebook that's failing -- ask to see it pasted
   verbatim. Don't assume it matches any known-good version, and don't assume it matches the one
   the diagnostic uses; read what's actually there.

If the diagnostic in Step 0 would not run, gather these by hand as well:

3. Which conda environment is active: `conda info --envs` (the active one is starred), and
   `python -c "import sys; print(sys.executable)"`.
4. The installed BuildStockQuery version:
   `python -c "from importlib.metadata import version; print(version('buildstock_query'))"`
   (Note: `buildstock_query.__version__` does not exist and will raise an AttributeError. That
   is not a sign of a broken install.)
5. Whether a named AWS profile is in use: check the `AWS_PROFILE` environment variable. If one
   is set, every `aws` command below needs `--profile <name>`, or it will read a different
   account than the notebook does.
6. The AWS CLI default region: `aws configure get region`. Note it, but be aware that
   BuildStockQuery always queries **us-west-2** unless told otherwise, and this project does not
   tell it otherwise. A CLI region other than us-west-2 is itself worth flagging, and it does not
   change where the model looks.
7. The Athena workgroup's configured output location -- always in us-west-2, regardless of what
   step 6 returned:

       aws athena get-work-group --work-group resstock-euss --region us-west-2

   Check whether the bucket it names belongs to this AWS account. (Compare against the `Account`
   field from `aws sts get-caller-identity`.)

Report the findings plainly before suggesting any fix. If something already looks wrong -- for
example, the output location isn't this account's own bucket -- say so and propose that as the
first thing to try, but confirm before assuming it's the whole problem.

## Step 2 -- Then troubleshoot one error at a time

1. Ask what error is showing up right now, when the grid-impact code is actually run -- not what
   showed up before. Ask for the full error message, pasted exactly.
2. Diagnose using the actual error text plus what Steps 0 and 1 found. Don't guess at an error
   that hasn't been shown.
3. Propose exactly one fix, in plain language, with the specific command or code change to make.
4. Ask what happened after trying it.
5. If a new error shows up, go back to step 2 with the new error. If it worked, confirm and stop.

Repeat this loop for as long as new errors keep appearing. There is no fixed number of rounds.

## Background on this project, for context only

This is the "grid impact" section of a residential energy model. It uses a library called
BuildStockQuery to pull electricity-usage data from Amazon Athena (a cloud database tool). Errors
in this area are almost always one of two things: a wrong argument in the `BuildStockQuery(...)`
call, or an AWS storage-location setting pointing at the wrong place. Only bring up a specific
past issue if the audit or the current error actually matches it -- don't assume history repeats.

**One known false lead.** If the Glue `upgrade` partition column comes up, AWS refuses to change
its type:

    InvalidInputException: Change of partitionColumn is not allowed on indexKey : upgrade

That refusal is correct and it blocks nothing. BuildStockQuery casts that column inside the SQL
it generates, so the column never needs to be an integer. Don't spend time on it, and don't try
to work around it by deleting and recreating the table.

## When it's resolved

Summarize in two or three plain sentences what was actually wrong and what fixed it, so there's a
short record to share afterward.
