// Brush stuff
// Brush Buttons and radius slider

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

RowLayout{

    ColumnLayout{
        Text{
            Layout.alignment: Qt.AlignHCenter
            text: "Type"
        }
        Layout.alignment: Qt.AlignCenter

        Button{
            Layout.preferredWidth: brushButton.width
            text: "1"
            onClicked: {
                obs.brush = 1
            }
        }
        Button{
            Layout.preferredWidth: brushButton.width
            text: "0"
            onClicked: {
                obs.brush = 0
            }
        }
        Button{
            id: brushButton
            text: "-1"
            onClicked: {
                obs.brush = -1
            }
        }

    }
    // Radius Slider
    ColumnLayout{
        Text{
            text: "Radius \n" + rSlider.value
        }
        Slider{
            id: rSlider
            value: obs.brushR
            orientation: Qt.Vertical
            // minimumValue: 1
            // maximumValue: 100
            from: 1
            to: 100
            stepSize: 1
            onValueChanged: {
                obs.brushR = value
            }
            onPressedChanged: {
                // show slider Hover when pressed, hide otherwise
                if( pressed )
                {

                }
                else {
                    Julia.newCirc()
                }
            }
        }
    }
}