param(
    [string]$EvaluationOutput = 'E:/infovis/generation/chabuduo/runs/distribution-evaluation-20261008-v3-reproduced'
)

& uv --cache-dir 'E:/infovis/generation/chabuduo/output/distribution-evaluation-runtime/uv-cache' `
    run --no-project `
    --python 'E:/infovis/generation/chabuduo/output/distribution-evaluation-runtime/.venv/Scripts/python.exe' `
    python 'E:/infovis/generation/chabuduo/runs/distribution-evaluation-20261008-v3/evaluate_generation_distributions.py' `
    --root 'E:/infovis/generation/chabuduo' `
    --inputs 'E:/infovis/generation/chabuduo/runs/minisky-preparation-20261003/inputs' `
    --output $EvaluationOutput --seed 42 --bootstrap 500
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
