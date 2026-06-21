# GrandOMR MuseScore Plugin Workflow

## Usage

Generate a plugin bundle while running GrandOMR.

For a PDF score:

```powershell
python run_score.py score.pdf 1 5 -o outputs/score.musicxml --plugin-output outputs/score_plugin
```

For a single image:

```powershell
python pipeline.py page.png -o outputs/page.musicxml --plugin-output outputs/page_plugin
```

Use a different `--plugin-output` directory for each recognition result. For example, if the MusicXML output is `outputs/beethoven_p9.musicxml`, use a matching bundle path such as `outputs/beethoven_p9_plugin`.

Start the viewer. If MuseScore is already in `PATH`, run:

```powershell
python grandomr_viewer.py outputs/page_plugin
```

Otherwise, pass your MuseScore executable path:

```powershell
python grandomr_viewer.py outputs/page_plugin --musescore "<path-to-MuseScore4.exe>"
```

This should automatically open a browser page. Leave it open for now; you do not need to click anything there yet.

Open the MusicXML file written by `-o` in MuseScore. For the examples above, open `outputs/score.musicxml` or `outputs/page.musicxml`.

Install the MuseScore plugin by copying:

```text
musescore_plugin/GrandOMR Plugin.qml
```

to your MuseScore 4 plugin directory. On Windows this is usually:

```text
%USERPROFILE%\Documents\MuseScore4\Plugins\GrandOMR Plugin.qml
```

In MuseScore, open `Plugins > Manage plugins`, find `GrandOMR Plugin`, and enable it. Then start it from `Plugins > Composing/arranging tools > GrandOMR Plugin`.

After the MuseScore plugin is running, wait until the browser viewer shows `MuseScore: Connected`. Do not start note selection or playback control while the viewer still says `Stale` or `Disconnected`.

Once connected, click a note in the browser viewer. MuseScore should select the matching note in the open score, add a red marker to it, and move the view to that page. If you open another score, click `Rescan Score` in the plugin.

## Method

The plugin workflow connects four things: the recognized MusicXML, the original page image, the browser viewer, and the MuseScore score window.

During recognition, GrandOMR keeps the HOMR notehead boxes from the page image and the trOMR note tokens used to write MusicXML. The custom MusicXML writer records each generated note token while writing XML notes. The pipeline then binds each XML note to a nearby HOMR notehead box in a local chord group and assigns a stable `omrId`.

The plugin output stores the page image, note boxes, and `omrId` metadata. `grandomr_viewer.py` reads that output, uses MuseScore CLI to inspect a tagged copy of the MusicXML, and builds a map from each `omrId` to the corresponding note object in MuseScore.

The browser viewer displays the page image and clickable GrandOMR note boxes. When a note is clicked, the viewer records the clicked `omrId`. The MuseScore QML plugin polls the viewer bridge, receives that `omrId`, selects the matching note in the open score, applies a temporary red marker, and moves the score view to the relevant page.
