// Analysis Panel

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform


ApplicationWindow {
    id: root
    width: 300 ; height: 150

    property int tb_width: 50

    ColumnLayout{
        Text {
            Layout.alignment: Qt.AlignHCenter
            text: qsTr("Temperature Sweep Analysis")
        }

        RowLayout{
            Layout.alignment: Qt.AlignHCenter

            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Start"
                }
                TextField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    id: tstart
                    validator: IntValidator {
                        bottom: 0; top: 100
                    }
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Step"
                }
                TextField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    id: tstep
                    validator: DoubleValidator {
                        bottom: 0.005 ; top: 10.0
                    }
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T End"
                }
                TextField{
                    Layout.alignment: Qt.AlignHCenter
                    width: tb_width
                    id: tend
                    validator: IntValidator {
                        bottom: 0; top: 100
                    }
                }
            }
        }
    }

}