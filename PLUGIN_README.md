# GrandOMR MuseScore Plugin Workflow

## Generate Plugin Output

For a PDF score:

```powershell
python run_score.py score.pdf 1 5 -o outputs/score.musicxml --plugin-output outputs/score_plugin
```

For a single image:

```powershell
python pipeline.py page.png -o outputs/page.musicxml --plugin-output outputs/page_plugin
```

The normal MusicXML output path is still written. `--plugin-output` additionally writes:

```text
score_plugin/
  score.musicxml
  manifest.json
  notes.json
  pages/
    page_0001.png
```

Open `score_plugin/score.musicxml` in MuseScore when using the viewer.

## Start Viewer/Bridge

```powershell
python grandomr_viewer.py outputs/score_plugin
```

This starts `http://127.0.0.1:8765/` and opens the browser viewer.

## MuseScore

The plugin file is:

```text
musescore_plugin/GrandOMR Plugin.qml
```

Copy it to:

```text
C:\Users\lcw\Documents\MuseScore4\Plugins\GrandOMR Plugin.qml
```

Then enable and run `GrandOMR Plugin` from MuseScore's Plugins menu. It automatically tries to connect to `http://127.0.0.1:8765`, scans the current score, polls for clicked notes, selects them, and nudges the MuseScore view to the target page.

Use `Rescan Score` after opening another score or after editing the score structure.
