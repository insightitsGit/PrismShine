# PrismShine comparative benchmark — Azure deployment (az CLI)
# Usage: pwsh bench/azure/deploy.ps1  (run from repo root)
# Creates: resource group, ACR (Basic), 3 ACI container groups. Prints endpoints.
$ErrorActionPreference = "Stop"

$LOC = "eastus"
$RG = "prismshine-bench-rg"
$SUFFIX = -join ((97..122) | Get-Random -Count 6 | ForEach-Object { [char]$_ })
$ACR = "prismshinebench$SUFFIX"

Write-Host "== resource group + ACR ($ACR) =="
az group create -n $RG -l $LOC -o none
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true -o none
$ACR_SERVER = az acr show -n $ACR --query loginServer -o tsv
$ACR_USER = az acr credential show -n $ACR --query username -o tsv
$ACR_PASS = az acr credential show -n $ACR --query "passwords[0].value" -o tsv

Write-Host "== cloud-building images (az acr build) =="
az acr build -r $ACR -t bench/prismshine:v1 -f bench/shims/prismshine/Dockerfile .
az acr build -r $ACR -t bench/hhem:v1 bench/shims/hhem
az acr build -r $ACR -t bench/ragas:v1 bench/shims/ragas
az acr build -r $ACR -t bench/ollama:v1 bench/shims/ollama

Write-Host "== container groups =="
az container create -g $RG -n shine-bench --image "$ACR_SERVER/bench/prismshine:v1" `
  --cpu 4 --memory 8 --ports 8000 --ip-address Public --dns-name-label "shine-bench-$SUFFIX" `
  --registry-login-server $ACR_SERVER --registry-username $ACR_USER --registry-password $ACR_PASS -o none

az container create -g $RG -n hhem-bench --image "$ACR_SERVER/bench/hhem:v1" `
  --cpu 4 --memory 8 --ports 8000 --ip-address Public --dns-name-label "hhem-bench-$SUFFIX" `
  --registry-login-server $ACR_SERVER --registry-username $ACR_USER --registry-password $ACR_PASS -o none

$yaml = Get-Content bench/azure/aci-ragas.yaml -Raw
$yaml = $yaml.Replace("__LOCATION__", $LOC).Replace("__NAME__", "ragas-bench")
$yaml = $yaml.Replace("__DNSLABEL__", "ragas-bench-$SUFFIX")
$yaml = $yaml.Replace("__ACR_SERVER__", $ACR_SERVER).Replace("__ACR_USER__", $ACR_USER).Replace("__ACR_PASS__", $ACR_PASS)
$tmp = New-TemporaryFile
Set-Content $tmp.FullName $yaml
az container create -g $RG --file $tmp.FullName -o none
Remove-Item $tmp.FullName

$SHINE = az container show -g $RG -n shine-bench --query ipAddress.fqdn -o tsv
$HHEM = az container show -g $RG -n hhem-bench --query ipAddress.fqdn -o tsv
$RAGAS = az container show -g $RG -n ragas-bench --query ipAddress.fqdn -o tsv

$targets = @{ "prismshine-fast" = "http://${SHINE}:8000"; "hhem" = "http://${HHEM}:8000"; "ragas" = "http://${RAGAS}:8000" }
$targets | ConvertTo-Json | Set-Content bench/runner/targets.json
Write-Host "targets.json written:"
Get-Content bench/runner/targets.json

Write-Host "`nRun:  python bench/runner/run_bench.py --targets bench/runner/targets.json --n 100 --b2 25 --ragas-limit 30 --out bench/runner/results/run1"
Write-Host "Teardown when done:  az group delete -n $RG --yes --no-wait"
