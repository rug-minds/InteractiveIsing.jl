import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

// Buttons at top of window
Item{
    width: childrenRect.width
    height: childrenRect.height

    ColumnLayout{
        spacing: 2
        // Init Graph Button
        Button {
            Layout.alignment: Qt.AlignCenter
            text: "Initialize Graph"
            onClicked: {
                Julia.initIsing()
            }
        }
        // Pause Simulation Button
        Button{
            Layout.alignment: Qt.AlignCenter
            text: {
                if(obs.isPaused)
                {
                    "Paused"
                }
                else{
                    "Running"
                }
            }
            onClicked: {
                Julia.toggleSimRunning()
            }
        }

        TextField{
            Layout.alignment: Qt.AlignCenter
            text: obs.layerName
            // Deselect when pressing elsewhere or enter
            Keys.onPressed: {
                if(event.key === Qt.Key_Return || event.key === Qt.Key_Enter)
                {
                    forceActiveFocus()
                }
            }
            onTextChanged: {
                Julia.setLayerName(text)
            }
            // onAccepted: {
            //     Julia.setLayerName(text)
            // }
        }
 
    }
}