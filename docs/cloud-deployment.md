# Optional Cloud Run deployment

**Prepared, not deployed.** These PowerShell instructions describe an optional Cloud Run setup for the Heating Load API.
No cloud commands were executed. They require an approved billing-enabled Google Cloud project,
the Google Cloud CLI, Docker, and permission to publish images and deploy a public service.
Storage and execution can incur charges, even with scale-to-zero. Run only after approval.

## Build and deploy

From the repository root, replace the project ID. Use a new, project-specific repository name;
if it already exists, confirm ownership before reusing it. Stop if any command fails.

```powershell
$ProjectId = "YOUR_APPROVED_PROJECT_ID"
$Region = "europe-west2"
$Repository = "heating-load-api"
$Service = "heating-load-api"
$Image = "${Region}-docker.pkg.dev/${ProjectId}/${Repository}/${Service}:v1"
gcloud auth login
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project $ProjectId
gcloud artifacts repositories create $Repository --repository-format docker --location $Region --project $ProjectId
gcloud auth configure-docker "${Region}-docker.pkg.dev"
docker build --platform linux/amd64 -t $Image .
docker push $Image
gcloud run deploy $Service --image $Image --project $ProjectId --region $Region --platform managed --port 8000 --cpu 1 --memory 512Mi --min 0 --max 1 --allow-unauthenticated
```

The container listens on port 8000; `--port 8000` makes Cloud Run route requests there.
This publishes a public prediction API. If your organization disallows public invocation,
ask its administrator rather than changing security policies.

## Verify and clean up

```powershell
$ServiceUrl = gcloud run services describe $Service --project $ProjectId --region $Region --format "value(status.url)"
Invoke-RestMethod "$ServiceUrl/health"
Invoke-RestMethod "$ServiceUrl/predict" -Method Post -ContentType "application/json" -Body (Get-Content examples/building.json -Raw)
```

Expect `status: ok` and one numeric `heating_load`. Save an actual screenshot and endpoint URL
before claiming a completed deployment. The local Docker screenshot is not cloud evidence.

When the service is no longer needed, delete only the service and dedicated repository created above:

```powershell
gcloud run services delete $Service --project $ProjectId --region $Region
gcloud artifacts repositories delete $Repository --project $ProjectId --location $Region
```

Check billing for remaining usage; deletion does not undo prior charges.
Reference: [Google Cloud container deployment guide](https://docs.cloud.google.com/run/docs/deploying).
