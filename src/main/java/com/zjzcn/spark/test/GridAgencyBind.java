package com.zjzcn.spark.test;

import net.dongliu.requests.RawResponse;
import net.dongliu.requests.Requests;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class GridAgencyBind {

    public static void main(String[] args) throws IOException {
        String url = "https://minos-web.ele.me/kael-webapi/grid/agency/bind/bind?gridId=%s&agencyId=%s";

        Map<String, String> cookies = new HashMap<>();
        cookies.put("COFFEE_TOKEN", "d428486c-4418-4e6a-9572-b84776650c48");

        Workbook wb = new HSSFWorkbook(
                GridAgencyBind.class.getClassLoader().getResourceAsStream("dwd.xls"));
        Sheet sheet = wb.getSheetAt(0);

        for (Row row : sheet) {
            try {
                String gridId = row.getCell(0).getStringCellValue();
                RawResponse resp = Requests.get(String.format(url, gridId, 14647795)).cookies(cookies).send();
                String respText = resp.readToText();
                int code = resp.getStatusCode();

                System.out.println("agency_id=14647795, " + "grid_id=" + gridId + ". code=" + code + ", " + respText);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        Workbook wb1 = new HSSFWorkbook(
                GridAgencyBind.class.getClassLoader().getResourceAsStream("crow.xls"));
        Sheet sheet1 = wb1.getSheetAt(0);

        for (Row row : sheet1) {
            try {
                String gridId = row.getCell(0).getStringCellValue();
                RawResponse resp = Requests.get(String.format(url, gridId, 14648410)).cookies(cookies).send();
                String respText = resp.readToText();

                System.out.println("agency_id=14648410, " + "grid_id=" + gridId + ". " + respText);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }
}
