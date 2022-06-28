//  Temperature Slider
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

Item{
    width: childrenRect.width
    height: childrenRect.height
    RowLayout{
        spacing: 2
        // Temp Slider
        Slider{
            height: 256
            value: obs.TIs
            orientation: Qt.Vertical
            // minimumValue: 0.0
            // maximumValue: 20
            from: 0.0
            to: 20.
            stepSize: 0.01
            onValueChanged: {
                obs.TIs = value
            }
        }
        // Temperature text
        Item{
            width: 32
            Text{
                Layout.alignment: Qt.AlignCenter
                text: qsTr("T=\n") + obs.TIs.toFixed(2)
            }
        }
    }
}