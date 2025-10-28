from noisy_speech_pipeline.triage_enrich import enrich_manifest_with_triage

if __name__ == "__main__":
    stats = enrich_manifest_with_triage()
    print(stats)
