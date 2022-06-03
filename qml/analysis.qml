// Analysis Panel  

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform


ApplicationWindow {
    id: root
    width: 100; height: 100

    Text {
        anchors.centerIn: parent
        text: qsTr("Hello World.")
    }
}