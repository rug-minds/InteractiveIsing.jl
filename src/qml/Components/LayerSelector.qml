import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

// Panels under plot
// Inserting Defects
// Magnetization
ColumnLayout{
    spacing: 2
    Text{
        Layout.alignment: Qt.AlignCenter
        text: "Layer" + " " + obs.activeLayer
    }
    RowLayout{
        Layout.alignment: Qt.AlignCenter
        Button{
            text: "<"
            onClicked:
            {
                if (obs.layer > 1){
                    obs.layer += -1
                }
            }
        }
        Button{
            Layout.alignment: Qt.AlignCenter
            text: ">"
            onClicked:
            {
                if (obs.layer < obs.nlayers){
                    obs.layer += 1
                }
            }
        }
    }
}