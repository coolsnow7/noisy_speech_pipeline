from datetime import datetime

from airflow.decorators import dag, task

from noisy_speech_pipeline.triage_enrich import enrich_manifest_with_triage


@dag(
    dag_id="triage_enrich",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["triage", "enrich"],
)
def triage_enrich():
    @task(task_id="enrich_manifest")
    def run():
        return enrich_manifest_with_triage()

    run()


dag = triage_enrich()
