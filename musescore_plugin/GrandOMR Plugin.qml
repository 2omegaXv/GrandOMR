import QtQuick
import QtQuick.Controls
import MuseScore 3.0

MuseScore {
    id: root

    version: "1.0"
    title: "GrandOMR Plugin"
    description: "Connect GrandOMR Viewer to the current MuseScore score."
    categoryCode: "composing-arranging-tools"
    pluginType: "dialog"
    width: 720
    height: 420

    property var noteIndex: []
    property int lastSequence: -1
    property bool requestInFlight: false
    property bool connected: false
    property string bridgeUrl: "http://127.0.0.1:8765"

    Column {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Label {
            text: "GrandOMR Plugin"
            font.bold: true
        }

        TextField {
            id: urlField
            width: parent.width
            text: root.bridgeUrl
            selectByMouse: true
            onEditingFinished: root.bridgeUrl = text
        }

        Row {
            spacing: 8
            Button {
                text: root.connected ? "Disconnect" : "Connect"
                onClicked: toggleConnect()
            }
            Button {
                text: "Rescan Score"
                onClicked: {
                    scanScore()
                    registerScore()
                }
            }
            Button {
                text: "Poll Once"
                onClicked: pollOnce()
            }
            Button {
                text: "Clear Log"
                onClicked: logBox.text = ""
            }
            Label {
                id: statusLabel
                text: "Idle"
                width: 300
                elide: Text.ElideRight
            }
        }

        TextArea {
            id: logBox
            width: parent.width
            height: 310
            readOnly: true
            wrapMode: TextArea.Wrap
        }
    }

    Timer {
        id: autoConnectTimer
        interval: 250
        repeat: false
        running: true
        onTriggered: {
            if (!root.connected) {
                toggleConnect()
            }
        }
    }

    Timer {
        id: pollTimer
        interval: 500
        repeat: true
        running: false
        onTriggered: pollOnce()
    }

    function log(message) {
        statusLabel.text = message
        logBox.text = new Date().toLocaleTimeString() + "  " + message + "\n" + logBox.text
    }

    function normalizeUrl(url) {
        while (url.length > 0 && url.charAt(url.length - 1) === "/") {
            url = url.substring(0, url.length - 1)
        }
        return url
    }

    function encodeQuery(value) {
        return encodeURIComponent(value === undefined || value === null ? "" : String(value))
    }

    function httpGet(path, callback, useRequestLock) {
        var shouldLock = useRequestLock === undefined ? true : useRequestLock
        if (shouldLock && requestInFlight) {
            return
        }
        if (shouldLock) {
            requestInFlight = true
        }
        var request = new XMLHttpRequest()
        var url = normalizeUrl(root.bridgeUrl) + path
        request.onreadystatechange = function() {
            if (request.readyState !== XMLHttpRequest.DONE) {
                return
            }
            if (shouldLock) {
                requestInFlight = false
            }
            if (request.status !== 200) {
                log("HTTP " + request.status + " from " + url)
                return
            }
            callback(request.responseText)
        }
        request.onerror = function() {
            if (shouldLock) {
                requestInFlight = false
            }
            log("HTTP error from " + url)
        }
        request.open("GET", url, true)
        request.send()
    }

    function toggleConnect() {
        root.bridgeUrl = urlField.text
        if (root.connected) {
            pollTimer.running = false
            root.connected = false
            log("Disconnected")
            return
        }
        scanScore()
        root.connected = true
        pollTimer.running = true
        log("Connected; polling " + root.bridgeUrl)
        registerScore()
        pollOnce()
    }

    function registerScore() {
        var scorePath = curScore ? curScore.filePath : ""
        httpGet("/register?scorePath=" + encodeQuery(scorePath)
                + "&noteCount=" + encodeQuery(noteIndex.length),
                function(text) {
                    log("Bridge register: " + text)
                })
    }

    function sendAck(sequence, ok, message) {
        httpGet("/selected?sequence=" + encodeQuery(sequence)
                + "&ok=" + encodeQuery(ok ? "true" : "false")
                + "&message=" + encodeQuery(message || ""),
                function(_text) {}, false)
    }

    function pollOnce() {
        if (!curScore) {
            log("No score is open")
            return
        }
        httpGet("/next?lastSequence=" + encodeQuery(lastSequence), function(text) {
            try {
                var response = JSON.parse(text)
                if (!response || !response.selector) {
                    return
                }
                handleSelector(response.selector)
            } catch (error) {
                log("Poll parse error: " + error)
            }
        })
    }

    function fractionToTicks(frac) {
        if (frac === undefined || frac === null) {
            return 0
        }
        if (typeof frac === "number") {
            return frac
        }
        if (frac.ticks !== undefined) {
            return Number(frac.ticks)
        }
        if (frac.numerator !== undefined && frac.denominator !== undefined) {
            return Math.round(480 * 4 * Number(frac.numerator) / Number(frac.denominator))
        }
        var text = String(frac)
        var parts = text.split("/")
        if (parts.length === 2) {
            var numerator = Number(parts[0])
            var denominator = Number(parts[1])
            if (!isNaN(numerator) && !isNaN(denominator) && denominator !== 0) {
                return Math.round(480 * 4 * numerator / denominator)
            }
        }
        var asNumber = Number(text)
        if (!isNaN(asNumber)) {
            return asNumber
        }
        return 0
    }

    function closeEnough(a, b) {
        return Math.abs(a - b) < 0.01
    }

    function fieldValue(selector, canonicalName, fallbackValue) {
        if (selector[canonicalName] !== undefined) {
            return selector[canonicalName]
        }
        if (canonicalName === "measureIdx" && selector.measuredlx !== undefined) {
            return selector.measuredlx
        }
        if (canonicalName === "voiceIdx" && selector.voiceldx !== undefined) {
            return selector.voiceldx
        }
        if (canonicalName === "noteIndex" && selector.notelndex !== undefined) {
            return selector.notelndex
        }
        return fallbackValue
    }

    function scanScore() {
        if (!curScore) {
            log("No score is open")
            return
        }

        var result = []
        var ntracks = curScore.ntracks || ((curScore.nstaves || 1) * 4)
        var maxMeasureSeen = 0

        for (var track = 0; track < ntracks; track++) {
            var cursor = curScore.newCursor()
            cursor.track = track
            cursor.rewind(0)
            var currentMeasureStartTick = null
            var measureIdx = -1

            while (cursor.segment) {
                var tick = fractionToTicks(cursor.tick)
                var measureStartTick = cursor.measure ? fractionToTicks(cursor.measure.tick) : Math.floor(tick / 1920) * 1920
                var measureTicks = cursor.measure ? fractionToTicks(cursor.measure.ticks) : 1920
                if (!measureTicks) {
                    measureTicks = 1920
                }
                if (currentMeasureStartTick === null || measureStartTick !== currentMeasureStartTick) {
                    measureIdx += 1
                    currentMeasureStartTick = measureStartTick
                }
                var beat = (tick - measureStartTick) / 480.0
                if (measureIdx > maxMeasureSeen) {
                    maxMeasureSeen = measureIdx
                }

                if (cursor.element && cursor.element.type === Element.CHORD) {
                    var staffIdx = Math.floor(track / 4)
                    var voiceIdx = track % 4
                    var pitchCounts = {}
                    for (var noteIndexInChord = 0; noteIndexInChord < cursor.element.notes.length; noteIndexInChord++) {
                        var note = cursor.element.notes[noteIndexInChord]
                        var samePitchIndex = pitchCounts[note.pitch] || 0
                        pitchCounts[note.pitch] = samePitchIndex + 1
                        result.push({
                            scanIndex: result.length,
                            staffIdx: staffIdx,
                            voiceIdx: voiceIdx,
                            track: track,
                            measureIdx: measureIdx,
                            beat: beat,
                            tick: tick,
                            fraction: cursor.element.fraction,
                            pitch: note.pitch,
                            noteIndex: noteIndexInChord,
                            samePitchIndex: samePitchIndex,
                            note: note
                        })
                    }
                }
                cursor.next()
            }
        }

        noteIndex = result
        log("Scanned notes=" + result.length
                + " measures=" + (maxMeasureSeen + 1)
                + " tracks=" + ntracks
                + " score=" + curScore.filePath)
    }

    function findScannedNote(selector) {
        if (noteIndex.length === 0) {
            scanScore()
        }

        var wantStaff = fieldValue(selector, "staffIdx", 0)
        var wantVoice = fieldValue(selector, "voiceIdx", 0)
        var wantMeasure = fieldValue(selector, "measureIdx", 0)
        var wantBeat = fieldValue(selector, "beat", 0)
        var wantNoteIndex = fieldValue(selector, "noteIndex", undefined)
        var fallbackSameBeat = null
        var fallbackNearestBeat = null
        var fallbackNearestDist = 999999

        for (var i = 0; i < noteIndex.length; i++) {
            var item = noteIndex[i]
            if (item.staffIdx !== wantStaff) {
                continue
            }
            if (item.measureIdx !== wantMeasure) {
                continue
            }
            if (selector.pitch !== undefined && item.pitch !== selector.pitch) {
                continue
            }
            if (item.voiceIdx === wantVoice
                    && closeEnough(item.beat, wantBeat)
                    && (wantNoteIndex === undefined || item.samePitchIndex === wantNoteIndex)) {
                return item
            }
            if (fallbackSameBeat === null && closeEnough(item.beat, wantBeat)) {
                fallbackSameBeat = item
            }
            var beatDist = Math.abs(item.beat - wantBeat)
            if (beatDist < fallbackNearestDist) {
                fallbackNearestDist = beatDist
                fallbackNearestBeat = item
            }
        }
        if (fallbackSameBeat !== null) {
            log("Fallback matched same beat; voice/noteIndex differed")
            return fallbackSameBeat
        }
        if (fallbackNearestBeat !== null && fallbackNearestDist <= 0.25) {
            log("Fallback matched nearest beat; delta=" + fallbackNearestDist)
            return fallbackNearestBeat
        }

        return null
    }

    function findScannedNoteStrict(selector) {
        if (noteIndex.length === 0) {
            scanScore()
        }

        var wantStaff = fieldValue(selector, "staffIdx", 0)
        var wantVoice = fieldValue(selector, "voiceIdx", 0)
        var wantMeasure = fieldValue(selector, "measureIdx", 0)
        var wantBeat = fieldValue(selector, "beat", 0)
        var wantNoteIndex = fieldValue(selector, "noteIndex", undefined)

        for (var i = 0; i < noteIndex.length; i++) {
            var item = noteIndex[i]
            if (item.staffIdx !== wantStaff || item.voiceIdx !== wantVoice) {
                continue
            }
            if (item.measureIdx !== wantMeasure || !closeEnough(item.beat, wantBeat)) {
                continue
            }
            if (selector.pitch !== undefined && item.pitch !== selector.pitch) {
                continue
            }
            if (wantNoteIndex !== undefined && item.samePitchIndex !== wantNoteIndex) {
                continue
            }
            return item
        }
        return null
    }

    function describeCandidates(selector) {
        var wantStaff = fieldValue(selector, "staffIdx", 0)
        var wantMeasure = fieldValue(selector, "measureIdx", 0)
        var wantPitch = selector.pitch
        var rows = []
        for (var i = 0; i < noteIndex.length; i++) {
            var item = noteIndex[i]
            if (item.staffIdx !== wantStaff) {
                continue
            }
            if (item.measureIdx !== wantMeasure) {
                continue
            }
            if (wantPitch !== undefined && item.pitch !== wantPitch) {
                continue
            }
            rows.push("m=" + item.measureIdx + " b=" + item.beat
                    + " v=" + item.voiceIdx + " p=" + item.pitch
                    + " ni=" + item.samePitchIndex)
            if (rows.length >= 8) {
                break
            }
        }
        if (rows.length === 0) {
            var sameStaffPitch = []
            var sameMeasure = []
            var sameStaff = []
            for (var j = 0; j < noteIndex.length; j++) {
                var cand = noteIndex[j]
                if (cand.staffIdx === wantStaff && cand.pitch === Number(wantPitch)) {
                    sameStaffPitch.push("m=" + cand.measureIdx + " b=" + cand.beat
                            + " v=" + cand.voiceIdx + " p=" + cand.pitch
                            + " ni=" + cand.samePitchIndex)
                    if (sameStaffPitch.length >= 8) {
                        break
                    }
                }
            }
            for (var k = 0; k < noteIndex.length && sameMeasure.length < 8; k++) {
                var mCand = noteIndex[k]
                if (mCand.staffIdx === wantStaff && mCand.measureIdx === wantMeasure) {
                    sameMeasure.push("m=" + mCand.measureIdx + " b=" + mCand.beat
                            + " v=" + mCand.voiceIdx + " p=" + mCand.pitch
                            + " ni=" + mCand.samePitchIndex)
                }
            }
            for (var s = 0; s < noteIndex.length && sameStaff.length < 8; s++) {
                var sCand = noteIndex[s]
                if (sCand.staffIdx === wantStaff) {
                    sameStaff.push("m=" + sCand.measureIdx + " b=" + sCand.beat
                            + " v=" + sCand.voiceIdx + " p=" + sCand.pitch
                            + " ni=" + sCand.samePitchIndex)
                }
            }
            return "no exact candidates on staff=" + wantStaff + " measure=" + wantMeasure
                    + " pitch=" + wantPitch
                    + " | same staff+pitch: " + (sameStaffPitch.length ? sameStaffPitch.join(" | ") : "<none>")
                    + " | same staff+measure: " + (sameMeasure.length ? sameMeasure.join(" | ") : "<none>")
                    + " | first same staff: " + (sameStaff.length ? sameStaff.join(" | ") : "<none>")
        }
        return rows.join(" | ")
    }

    function focusMatchedLocation(match) {
        try {
            var cursor = curScore.newCursor()
            cursor.track = match.track
            if (match.fraction) {
                cursor.rewindToFraction(match.fraction)
            }
        } catch (error) {
            log("Focus rewind failed: " + error)
        }

        var commandsText = "next-chord,prev-chord"
        if (match.scanIndex > 0 && match.scanIndex >= noteIndex.length - 1) {
            commandsText = "prev-chord,next-chord"
        }
        var commands = commandsText.split(",")
        for (var i = 0; i < commands.length; i++) {
            try {
                cmd(commands[i])
            } catch (error2) {
                log("Focus cmd failed " + commands[i] + ": " + error2)
            }
        }
    }

    function handleSelector(selector) {
        var sequence = selector.sequence || 0
        if (sequence === lastSequence) {
            return
        }
        lastSequence = sequence
        var match = findScannedNote(selector)
        if (!match) {
            scanScore()
            match = findScannedNote(selector)
        }
        if (!match) {
            var msg = "No match for " + JSON.stringify(selector) + " candidates: " + describeCandidates(selector)
            log(msg)
            sendAck(sequence, false, msg)
            return
        }

        curScore.startCmd("GrandOMR select note")
        curScore.selection.clear()
        curScore.selection.select(match.note, false)
        curScore.endCmd()
        focusMatchedLocation(match)
        curScore.startCmd("GrandOMR restore note selection")
        curScore.selection.clear()
        curScore.selection.select(match.note, false)
        curScore.endCmd()
        var okMsg = "Selected seq=" + sequence
                + " m=" + match.measureIdx
                + " b=" + match.beat
                + " pitch=" + match.pitch
        log(okMsg)
        sendAck(sequence, true, okMsg)
    }
}
