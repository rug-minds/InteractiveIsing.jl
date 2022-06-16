// Analysis Panel

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform


ApplicationWindow {
    id: root
    width: 500 ; height: 300

    property int tb_width: 50

    ColumnLayout{
        anchors.centerIn: parent
        Text {
            Layout.alignment: Qt.AlignHCenter
            text: qsTr("Temperature Sweep Analysis")
        }

        // Temperature parameters
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "T Start"
                }

                TextField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
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

        // Number of datapoints
        ColumnLayout{
            Text{
                Layout.alignment: Qt.AlignHCenter
                text: "Number of datapoints for every temperature"
            }

            TextField{
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                id: dpoints
                validator: IntValidator {
                    bottom: 0; top: 100
                }
            }

        }


        // Waiting times
        Text{
            Layout.alignment: Qt.AlignHCenter
            text: "Waiting Times"
        }
        RowLayout{
            Layout.alignment: Qt.AlignHCenter
            spacing: 8
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between datapoints"
                }

                TextField{
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    id: dpointwait
                    validator: IntValidator {
                        bottom: 1; top: 100
                    }
                }

            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Between Temperatures"
                }
                TextField{
                    Layout.alignment: Qt.AlignHCenter
                    id: stepwait
                    validator: IntValidator {
                        bottom: 0 ; top: 100
                    }
                }
            }
            ColumnLayout{
                Text{
                    Layout.alignment: Qt.AlignHCenter
                    text: "Wait for equilibration"
                }
                TextField{
                    Layout.alignment: Qt.AlignHCenter
                    id: equiwait
                    validator: IntValidator {
                        bottom: 0; top: 100
                    }
                }
            }
        }

        Button{
            Layout.alignment: Qt.AlignHCenter
            text: "Start Analysis"
            onClicked: {
                Julia.println(parseInt(dpoints.value))
                close()
            }
        }
    }

}
