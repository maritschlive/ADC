# ADC Workspace Uebersicht

Diese Datei dokumentiert die aktuelle Struktur im Workspace und erklaert kurz,
was jeder sichtbare Ordner bzw. jede sichtbare Datei macht.

Hinweise:
- In den Bild-/Maskenordnern liegen sehr viele numerierte PNG-Dateien.
  Diese werden als Dateigruppen beschrieben (statt jede einzelne Bilddatei aufzulisten).
- Die Beschreibungen sind bewusst kurz und praxisorientiert.

## Baumstruktur mit Kurzbeschreibung

<pre>
<span style="color:#1F6FEB;font-weight:600;">ADC/</span>
|-- <span style="color:#1F6FEB;">.git/</span> - Lokale Git-Metadaten (Branches, History, Index).
|-- <span style="color:#2DA44E;">.gitignore</span> - Regeln, welche Dateien/Ordner Git ignorieren soll.
|-- <span style="color:#1F6FEB;">.venv/</span> - Lokale Python-Virtual-Environment (Interpreter + installierte Pakete).
|-- <span style="color:#1F6FEB;">__pycache__/</span> - Von Python erzeugter Bytecode-Cache.
|-- <span style="color:#2DA44E;">NEXT_STEPS_PLAN.md</span> - Operativer Retry-/Monitoring-Plan nach Run-Analyse.
|-- <span style="color:#2DA44E;">PROJECT_TRAINING_GUIDE.md</span> - Detailliertes Betriebs-Handbuch fuer Setup, Training, Slurm, Auswertung.
|-- <span style="color:#2DA44E;">README.md</span> - Projektueberblick, Quickstart und Kontext zum ADC-Fork.
|-- <span style="color:#2DA44E;">analyze_runs.py</span> - Analysiert runs/ und erzeugt trainingsbezogenen Markdown-Report.
|-- <span style="color:#2DA44E;">config.py</span> - Schalter fuer globale Laufzeitoptionen (z. B. save_memory).
|-- <span style="color:#2DA44E;">create_control_ckpt.py</span> - Erzeugt control_sd15.ckpt aus SD-v1.5-Gewichten.
|-- <span style="color:#2DA44E;">create_liver_sample.py</span> - Baut ein synthetisches Leber-Maskenbeispiel fuer Inferenzdemos.
|-- <span style="color:#2DA44E;">create_sample_data.py</span> - Erstellt Demo-Maske/-Bild und demo_prompt fuer schnelle Tests.
|-- <span style="color:#2DA44E;">download_weights.py</span> - Laedt SD-v1.5- und ADC-Gewichte von Hugging Face herunter.
|-- <span style="color:#2DA44E;">environment.yaml</span> - Legacy Conda/Pip-Umgebungsspezifikation.
|-- <span style="color:#2DA44E;">evaluate_adc.py</span> - Bewertet generierte Bilder mit FID, SSIM und LPIPS.
|-- <span style="color:#2DA44E;">prepare_liver_data.py</span> - Konvertiert Rohdaten in ADC-Datenformat (train/val/test + prompt.json).
|-- <span style="color:#2DA44E;">pyproject.toml</span> - Projektmetadaten und Python-Abhaengigkeiten (uv/PEP 621).
|-- <span style="color:#2DA44E;">run_all.py</span> - Orchestriert mehrere Training-Presets sequentiell inkl. Dependency-Logik.
|-- <span style="color:#2DA44E;">segmentation_integration.py</span> - Prototypen fuer ADC+Segmentierungsmodell-Integration.
|-- <span style="color:#2DA44E;">setup_adc.py</span> - One-command Setup (Dependencies, Gewichte, Control-Checkpoint).
|-- <span style="color:#2DA44E;">share.py</span> - Zentraler Hook, der Konfig-Hacks (Verbosity, Sliced Attention) aktiviert.
|-- <span style="color:#2DA44E;">tool_add_control.py</span> - Urspruengliches Tool zum Initialisieren eines ControlNet-Checkpoints.
|-- <span style="color:#2DA44E;">tool_merge_control.py</span> - Merged SD-Basis und trainierte Control-Gewichte in eine Datei.
|-- <span style="color:#2DA44E;">tool_transfer_control.py</span> - Uebertraegt Control-Offsets auf ein anderes Basismodell.
|-- <span style="color:#2DA44E;">tutorial_dataset.py</span> - Standard-Trainingsdataset-Loader (Mask->Hint, Bild->Target).
|-- <span style="color:#2DA44E;">tutorial_dataset_sample.py</span> - Dataset-Loader fuer Inferenz/Test-Promptdateien.
|-- <span style="color:#2DA44E;">tutorial_dataset_test.py</span> - Mini-Checkskript fuer Dataset-Loader und Tensorformen.
|-- <span style="color:#2DA44E;">tutorial_inference.py</span> - Originale CUDA-zentrierte Inferenzpipeline.
|-- <span style="color:#2DA44E;">tutorial_inference_local.py</span> - Lokale Inferenz (MPS/CPU/CUDA) mit Ergebnisexport.
|-- <span style="color:#2DA44E;">tutorial_train.py</span> - Originales Multi-GPU/DeepSpeed-Trainingsskript.
|-- <span style="color:#2DA44E;">tutorial_train_single_gpu.py</span> - Haupt-Trainingsskript mit Presets und Hardware-Schaltern.
|-- <span style="color:#2DA44E;">uv.lock</span> - Lockfile fuer reproduzierbare Python-Abhaengigkeiten via uv.
|-- <span style="color:#2DA44E;">vram_calculator.py</span> - Schaetzt VRAM-Bedarf pro Preset anhand realer Modellteile.
|
|-- <span style="color:#1F6FEB;">cldm/</span> - ADC-spezifische Modell- und Sampling-Erweiterungen auf LDM-Basis.
|   |-- <span style="color:#2DA44E;">cldm.py</span> - Kernlogik von ControlLDM/ControlNet, Forward/Loss/Optimierergruppierung.
|   |-- <span style="color:#2DA44E;">ddim_hacked.py</span> - Angepasster DDIM-Sampler (ControlNet-spezifische Samplingpfade).
|   |-- <span style="color:#2DA44E;">dhi.py</span> - FeatureExtractor fuer Hint-Verarbeitung (Masken-Feature-Pipeline).
|   |-- <span style="color:#2DA44E;">hack.py</span> - Runtime-Hacks (z. B. CLIP-Forward, sliced attention, logging quiet).
|   |-- <span style="color:#2DA44E;">logger.py</span> - Lightning-Callback fuer Trainingsbild-Logging und Blended-Previews.
|   |-- <span style="color:#2DA44E;">model.py</span> - Modellfabrik und Checkpoint-State-Dict-Ladefunktionen.
|
|-- <span style="color:#1F6FEB;">data/</span> - Projekt-Datenbereich fuer Prompts, Splits und Bild-/Maskenpaare.
|   |-- <span style="color:#2DA44E;">prompt.json</span> - Kombinierte Promptliste (globaler Datensatzindex).
|   |-- <span style="color:#1F6FEB;">dataset/</span> - Zusatz-/Referenzdatensatzstruktur.
|   |   |-- <span style="color:#2DA44E;">dataset.yaml</span> - Beispiel-/Legacy-Datasetdefinition (Pfade/Klassen).
|   |   |-- <span style="color:#1F6FEB;">images/</span> - Bilddateien fuer diesen Dataset-Unterordner.
|   |   |-- <span style="color:#1F6FEB;">labels/</span> - Labeldateien (z. B. Segmentklassen/Annotationen).
|   |   |-- <span style="color:#1F6FEB;">masks/</span> - Maskendateien zum Dataset-Unterordner.
|   |-- <span style="color:#1F6FEB;">train/</span> - Trainingssplit.
|   |   |-- <span style="color:#2DA44E;">prompt.json</span> - Prompt-Mapping fuer Trainingsdaten.
|   |   |-- <span style="color:#1F6FEB;">images/</span> - Train-RGB-Bilder (nummerierte PNGs).
|   |   |-- <span style="color:#1F6FEB;">masks/</span> - Train-Masken (nummerierte, binaere PNGs).
|   |-- <span style="color:#1F6FEB;">val/</span> - Validierungssplit.
|   |   |-- <span style="color:#2DA44E;">prompt.json</span> - Prompt-Mapping fuer Validierungsdaten.
|   |   |-- <span style="color:#1F6FEB;">images/</span> - Val-RGB-Bilder (nummerierte PNGs).
|   |   |-- <span style="color:#1F6FEB;">masks/</span> - Val-Masken (nummerierte, binaere PNGs).
|   |-- <span style="color:#1F6FEB;">test/</span> - Test-/Inferenzsplit.
|       |-- <span style="color:#2DA44E;">prompt.json</span> - Prompt-Mapping fuer Testdaten.
|       |-- <span style="color:#1F6FEB;">images/</span> - Test-RGB-Bilder (nummerierte PNGs).
|       |-- <span style="color:#1F6FEB;">masks/</span> - Test-Masken (nummerierte, binaere PNGs).
|
|-- <span style="color:#1F6FEB;">examples/</span> - Kleine Beispieloutputs fuer README/Demozwecke.
|   |-- <span style="color:#2DA44E;">liver_10step.png</span> - Beispielbild aus 10-DDIM-Schritten (Lebermaske).
|   |-- <span style="color:#2DA44E;">liver_mask.png</span> - Beispiel-Lebermaske.
|   |-- <span style="color:#2DA44E;">polyp_10step.png</span> - Beispielbild aus 10-DDIM-Schritten (Polypmaske).
|   |-- <span style="color:#2DA44E;">polyp_mask.png</span> - Beispiel-Polypmaske.
|
|-- <span style="color:#1F6FEB;">images/</span> - Abbildungen fuer README/Projektdokumentation.
|   |-- <span style="color:#2DA44E;">figure1.png</span> - Hauptarchitektur/Overview-Abbildung.
|   |-- <span style="color:#2DA44E;">figure2.png</span> - Konvergenz-/Vergleichsvisualisierung.
|   |-- <span style="color:#2DA44E;">figure3.png</span> - Visualisierungsbeispiel (Kidney/CT).
|   |-- <span style="color:#2DA44E;">figure4.png</span> - Visualisierungsbeispiel (Polyp/RGB).
|
|-- <span style="color:#1F6FEB;">ldm/</span> - Upstream Latent-Diffusion-Basiscode.
|   |-- <span style="color:#2DA44E;">util.py</span> - Allgemeine Hilfsfunktionen (Logging, Instanziierung, Optimizer-Helfer).
|   |-- <span style="color:#1F6FEB;">data/</span>
|   |   |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer ldm.data.
|   |   |-- <span style="color:#2DA44E;">util.py</span> - Datennahe Utility-Funktionen.
|   |-- <span style="color:#1F6FEB;">models/</span>
|   |   |-- <span style="color:#2DA44E;">autoencoder.py</span> - VAE/AutoencoderKL Implementierung.
|   |   |-- <span style="color:#1F6FEB;">diffusion/</span>
|   |       |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer Diffusionsmodelle.
|   |       |-- <span style="color:#2DA44E;">ddim.py</span> - DDIM-Sampler (MPS-/Device-kompatible Variante).
|   |       |-- <span style="color:#2DA44E;">ddpm.py</span> - DDPM/LDM-Trainingskern, Schedules, Samplinglogik.
|   |       |-- <span style="color:#2DA44E;">plms.py</span> - PLMS-Sampler fuer schnelle Diffusionsinferenz.
|   |       |-- <span style="color:#2DA44E;">sampling_util.py</span> - Hilfsfunktionen fuer Sampling/Schedules.
|   |       |-- <span style="color:#1F6FEB;">dpm_solver/</span>
|   |           |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer DPM-Solver.
|   |           |-- <span style="color:#2DA44E;">dpm_solver.py</span> - Numerischer Solver fuer schnelle Diffusionsschritte.
|   |           |-- <span style="color:#2DA44E;">sampler.py</span> - Sampler-Adapter um DPM-Solver.
|   |-- <span style="color:#1F6FEB;">modules/</span>
|       |-- <span style="color:#2DA44E;">attention.py</span> - Attention-/Transformer-Bausteine fuer UNet/LDM.
|       |-- <span style="color:#2DA44E;">ema.py</span> - Exponential-Moving-Average-Helfer fuer Modellgewichte.
|       |-- <span style="color:#1F6FEB;">diffusionmodules/</span>
|       |   |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer Diffusionsmodule.
|       |   |-- <span style="color:#2DA44E;">model.py</span> - Encoder/Decoder-Bausteine fuer Diffusionsarchitektur.
|       |   |-- <span style="color:#2DA44E;">openaimodel.py</span> - UNet-Implementierung im OpenAI-Diffusionstil.
|       |   |-- <span style="color:#2DA44E;">upscaling.py</span> - Komponenten fuer Upscaling-Pfade.
|       |   |-- <span style="color:#2DA44E;">util.py</span> - Utility-Funktionen fuer Zeitschritte/Noise/ResBlocks.
|       |-- <span style="color:#1F6FEB;">distributions/</span>
|       |   |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer Verteilungsfunktionen.
|       |   |-- <span style="color:#2DA44E;">distributions.py</span> - Gauss-Verteilungen, KL, Sampling-Helfer.
|       |-- <span style="color:#1F6FEB;">encoders/</span>
|       |   |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer Encoder.
|       |   |-- <span style="color:#2DA44E;">modules.py</span> - Text-/Bild-Encoder (u. a. CLIP Wrapper).
|       |-- <span style="color:#1F6FEB;">image_degradation/</span>
|       |   |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer Degradation-Utilities.
|       |   |-- <span style="color:#2DA44E;">bsrgan.py</span> - BSRGAN-Degradationspipeline.
|       |   |-- <span style="color:#2DA44E;">bsrgan_light.py</span> - Vereinfachte/leichtere BSRGAN-Variante.
|       |   |-- <span style="color:#2DA44E;">utils_image.py</span> - Bildhelper fuer Degradation/Preprocessing.
|       |   |-- <span style="color:#1F6FEB;">utils/</span>
|       |       |-- <span style="color:#2DA44E;">test.png</span> - Test-/Beispielbild fuer Degradation-Utilities.
|       |-- <span style="color:#1F6FEB;">midas/</span>
|           |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer MiDaS-Integration.
|           |-- <span style="color:#2DA44E;">api.py</span> - API-Helfer fuer Tiefenschaetzung.
|           |-- <span style="color:#2DA44E;">utils.py</span> - Utility-Funktionen fuer MiDaS-Modelle.
|           |-- <span style="color:#1F6FEB;">midas/</span>
|               |-- <span style="color:#2DA44E;">__init__.py</span> - Paketmarker fuer interne MiDaS-Module.
|               |-- <span style="color:#2DA44E;">base_model.py</span> - Basisklassen fuer MiDaS-Modelle.
|               |-- <span style="color:#2DA44E;">blocks.py</span> - Architekturbausteine der MiDaS-Netze.
|               |-- <span style="color:#2DA44E;">dpt_depth.py</span> - DPT-Tiefenmodell.
|               |-- <span style="color:#2DA44E;">midas_net.py</span> - Klassisches MiDaS-Netz.
|               |-- <span style="color:#2DA44E;">midas_net_custom.py</span> - Angepasste MiDaS-Variante.
|               |-- <span style="color:#2DA44E;">transforms.py</span> - Bildtransforms fuer MiDaS-Pipeline.
|               |-- <span style="color:#2DA44E;">vit.py</span> - Vision-Transformer-Komponenten fuer MiDaS.
|
|-- <span style="color:#1F6FEB;">logs/</span> - Slurm Joblogs (stdout/stderr) fuer Setup/Training.
|   |-- <span style="color:#2DA44E;">.gitkeep</span> - Haelt den Ordner in Git, auch wenn sonst leer.
|   |-- <span style="color:#2DA44E;">adc_setup_711.err</span> - STDERR des Setup-Jobs 711.
|   |-- <span style="color:#2DA44E;">adc_setup_711.out</span> - STDOUT des Setup-Jobs 711.
|   |-- <span style="color:#2DA44E;">adc_setup_713.err</span> - STDERR des Setup-Jobs 713.
|   |-- <span style="color:#2DA44E;">adc_setup_713.out</span> - STDOUT des Setup-Jobs 713.
|   |-- <span style="color:#2DA44E;">adc_setup_714.err</span> - STDERR des Setup-Jobs 714.
|   |-- <span style="color:#2DA44E;">adc_setup_714.out</span> - STDOUT des Setup-Jobs 714.
|   |-- <span style="color:#2DA44E;">adc_setup_715.err</span> - STDERR des Setup-Jobs 715.
|   |-- <span style="color:#2DA44E;">adc_setup_715.out</span> - STDOUT des Setup-Jobs 715.
|   |-- <span style="color:#2DA44E;">adc_train_712.err</span> - STDERR des Training-Jobs 712.
|   |-- <span style="color:#2DA44E;">adc_train_712.out</span> - STDOUT des Training-Jobs 712.
|   |-- <span style="color:#2DA44E;">adc_train_716.err</span> - STDERR des Training-Jobs 716.
|   |-- <span style="color:#2DA44E;">adc_train_716.out</span> - STDOUT des Training-Jobs 716.
|   |-- <span style="color:#2DA44E;">adc_train_717.err</span> - STDERR des Training-Jobs 717.
|   |-- <span style="color:#2DA44E;">adc_train_717.out</span> - STDOUT des Training-Jobs 717.
|   |-- <span style="color:#2DA44E;">adc_train_718.err</span> - STDERR des Training-Jobs 718.
|   |-- <span style="color:#2DA44E;">adc_train_718.out</span> - STDOUT des Training-Jobs 718.
|   |-- <span style="color:#2DA44E;">adc_train_719.err</span> - STDERR des Training-Jobs 719.
|   |-- <span style="color:#2DA44E;">adc_train_719.out</span> - STDOUT des Training-Jobs 719.
|   |-- <span style="color:#2DA44E;">adc_train_720.err</span> - STDERR des Training-Jobs 720.
|   |-- <span style="color:#2DA44E;">adc_train_720.out</span> - STDOUT des Training-Jobs 720.
|   |-- <span style="color:#2DA44E;">adc_train_721.err</span> - STDERR des Training-Jobs 721.
|   |-- <span style="color:#2DA44E;">adc_train_721.out</span> - STDOUT des Training-Jobs 721.
|   |-- <span style="color:#2DA44E;">adc_train_722.err</span> - STDERR des Training-Jobs 722.
|   |-- <span style="color:#2DA44E;">adc_train_722.out</span> - STDOUT des Training-Jobs 722.
|   |-- <span style="color:#2DA44E;">adc_train_723.err</span> - STDERR des Training-Jobs 723.
|   |-- <span style="color:#2DA44E;">adc_train_723.out</span> - STDOUT des Training-Jobs 723.
|   |-- <span style="color:#2DA44E;">adc_train_724.err</span> - STDERR des Training-Jobs 724.
|   |-- <span style="color:#2DA44E;">adc_train_724.out</span> - STDOUT des Training-Jobs 724.
|   |-- <span style="color:#2DA44E;">adc_train_725.err</span> - STDERR des Training-Jobs 725.
|   |-- <span style="color:#2DA44E;">adc_train_725.out</span> - STDOUT des Training-Jobs 725.
|   |-- <span style="color:#2DA44E;">adc_train_726.err</span> - STDERR des Training-Jobs 726.
|   |-- <span style="color:#2DA44E;">adc_train_726.out</span> - STDOUT des Training-Jobs 726.
|   |-- <span style="color:#2DA44E;">adc_train_727.err</span> - STDERR des Training-Jobs 727.
|   |-- <span style="color:#2DA44E;">adc_train_727.out</span> - STDOUT des Training-Jobs 727.
|   |-- <span style="color:#2DA44E;">adc_train_728.err</span> - STDERR des Training-Jobs 728.
|   |-- <span style="color:#2DA44E;">adc_train_728.out</span> - STDOUT des Training-Jobs 728.
|   |-- <span style="color:#2DA44E;">adc_train_729.err</span> - STDERR des Training-Jobs 729.
|   |-- <span style="color:#2DA44E;">adc_train_729.out</span> - STDOUT des Training-Jobs 729.
|   |-- <span style="color:#2DA44E;">adc_train_730.err</span> - STDERR des Training-Jobs 730.
|   |-- <span style="color:#2DA44E;">adc_train_730.out</span> - STDOUT des Training-Jobs 730.
|   |-- <span style="color:#2DA44E;">adc_train_731.err</span> - STDERR des Training-Jobs 731.
|   |-- <span style="color:#2DA44E;">adc_train_731.out</span> - STDOUT des Training-Jobs 731.
|   |-- <span style="color:#2DA44E;">adc_train_732.err</span> - STDERR des Training-Jobs 732.
|   |-- <span style="color:#2DA44E;">adc_train_732.out</span> - STDOUT des Training-Jobs 732.
|   |-- <span style="color:#2DA44E;">adc_train_733.err</span> - STDERR des Training-Jobs 733.
|   |-- <span style="color:#2DA44E;">adc_train_733.out</span> - STDOUT des Training-Jobs 733.
|   |-- <span style="color:#2DA44E;">adc_train_734.err</span> - STDERR des Training-Jobs 734.
|   |-- <span style="color:#2DA44E;">adc_train_734.out</span> - STDOUT des Training-Jobs 734.
|   |-- <span style="color:#2DA44E;">adc_train_735.err</span> - STDERR des Training-Jobs 735.
|   |-- <span style="color:#2DA44E;">adc_train_735.out</span> - STDOUT des Training-Jobs 735.
|   |-- <span style="color:#2DA44E;">adc_train_736.err</span> - STDERR des Training-Jobs 736.
|   |-- <span style="color:#2DA44E;">adc_train_736.out</span> - STDOUT des Training-Jobs 736.
|
|-- <span style="color:#1F6FEB;">models/</span>
|   |-- <span style="color:#2DA44E;">cldm_v15.yaml</span> - Modellkonfiguration fuer ControlLDM/ControlNet-Architektur.
|
|-- <span style="color:#1F6FEB;">papers/</span> - Referenzpublikationen (PDFs) zum methodischen Hintergrund.
|   |-- <span style="color:#2DA44E;">2010.02502v4_DDIM.pdf</span> - DDIM-Paper (Samplingbeschleunigung).
|   |-- <span style="color:#2DA44E;">2112.10752v2_LDM_StableDiffusion.pdf</span> - Latent Diffusion Grundlage.
|   |-- <span style="color:#2DA44E;">2207.12598v1_ClassifierFreeGuidance.pdf</span> - Classifier-Free Guidance Theorie.
|   |-- <span style="color:#2DA44E;">2302.05543v3_ControlNet.pdf</span> - ControlNet-Architekturpaper.
|   |-- <span style="color:#2DA44E;">2505.06068v1_SiameseDiffusion.pdf</span> - Siamese-Diffusion Referenz.
|   |-- <span style="color:#2DA44E;">2507.23652v1_ADC.pdf</span> - ADC-Hauptpaper.
|
|-- <span style="color:#1F6FEB;">runs/</span> - Experimentordner mit Lightning-Logs pro Run/Tag.
|   |-- <span style="color:#1F6FEB;">paper_faithful_scratch_20260419_233156_full_paper_scratch/</span>
|   |   |-- <span style="color:#1F6FEB;">sanity/</span> - Sanity-Checks/Initialisierungsoutputs.
|   |   |-- <span style="color:#1F6FEB;">tb/</span> - TensorBoard Event-Dateien.
|   |   |-- <span style="color:#1F6FEB;">version_0/</span> - Lightning-Version mit Checkpoints/Metriken.
|   |-- <span style="color:#1F6FEB;">polyp_transfer_20260419_233156_full_transfer_base/</span>
|   |   |-- <span style="color:#1F6FEB;">sanity/</span> - Sanity-Checks/Initialisierungsoutputs.
|   |   |-- <span style="color:#1F6FEB;">tb/</span> - TensorBoard Event-Dateien.
|   |   |-- <span style="color:#1F6FEB;">version_0/</span> - Lightning-Version mit Checkpoints/Metriken.
|   |-- <span style="color:#1F6FEB;">scratch/</span>
|   |   |-- <span style="color:#1F6FEB;">sanity/</span> - Sanity-Checks/Initialisierungsoutputs.
|   |   |-- <span style="color:#1F6FEB;">tb/</span> - TensorBoard Event-Dateien.
|   |   |-- <span style="color:#1F6FEB;">version_0/</span> - Lightning-Version mit Checkpoints/Metriken.
|   |-- <span style="color:#1F6FEB;">scratch_20260419_233156_full_scratch_base/</span>
|   |   |-- <span style="color:#1F6FEB;">sanity/</span> - Sanity-Checks/Initialisierungsoutputs.
|   |   |-- <span style="color:#1F6FEB;">tb/</span> - TensorBoard Event-Dateien.
|   |   |-- <span style="color:#1F6FEB;">version_0/</span> - Lightning-Version mit Checkpoints/Metriken.
|   |-- <span style="color:#1F6FEB;">scratch_unlocked_20260419_233156_full_scratch_unlocked/</span>
|       |-- <span style="color:#1F6FEB;">sanity/</span> - Sanity-Checks/Initialisierungsoutputs.
|       |-- <span style="color:#1F6FEB;">tb/</span> - TensorBoard Event-Dateien.
|       |-- <span style="color:#1F6FEB;">version_0/</span> - Lightning-Version mit Checkpoints/Metriken.
|
|-- <span style="color:#1F6FEB;">slurm/</span> - Batchskripte fuer Setup, Training, Inferenz und Sweeps.
|   |-- <span style="color:#2DA44E;">setup.sh</span> - Einmaliges Cluster-Setup (uv + setup_adc.py).
|   |-- <span style="color:#2DA44E;">train.sh</span> - Haupt-Trainingsjob (ein Preset oder alle via run_all.py).
|   |-- <span style="color:#2DA44E;">train_all.sh</span> - Komfort-Submit fuer alle Presets (single oder split mode).
|   |-- <span style="color:#2DA44E;">infer.sh</span> - Batch-Inferenzjob mit tutorial_inference_local.py.
|   |-- <span style="color:#2DA44E;">submit_experiments.sh</span> - Profilbasierte Sweep-Submits mit Dependencies.
</pre>
