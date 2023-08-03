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
        text: "Layer" + " " + obs.layerIdx + "/" + obs.nlayers
    }
    RowLayout{
        Layout.alignment: Qt.AlignCenter
        Button{
            text: "<"
            onClicked:
            {
                Julia.changeLayer(-1)
            }
        }
        Button{
            Layout.alignment: Qt.AlignCenter
            text: ">"
            onClicked:
            {
                Julia.changeLayer(1)
            }
        }
    }
}