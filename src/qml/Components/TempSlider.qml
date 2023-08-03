//  Temperature Slider
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

Rectangle{
    property var tWidth: 32
    width: tWidth
    color: "transparent"
    
    // height: childrenRect.height
    height: 512
    Column{
        // Layout.alignment: Qt.AlignCenter
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 2

        // Temperature text
        Item{
            // color: "transparent"
            // anchors.bottom: tslider.top
            anchors.horizontalCenter: parent.horizontalCenter
            width: tWidth
            height: childrenRect.height
            Text{
                anchors.horizontalCenter: parent.horizontalCenter
                text: qsTr("T=\n") + tslider.value.toFixed(2)
            }
        }

        // Temp Slider
        Slider{
            id: tslider
            // anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            height: 200
            value: obs.Temp
            orientation: Qt.Vertical
            // minimumValue: 0.0
            // maximumValue: 20
            from: 0.0
            to: 20.
            stepSize: 0.01
            onValueChanged: {
                obs.Temp = value
            }
        }

        // Button{
        //     text: "Anneal"
            
        // }

    }
}